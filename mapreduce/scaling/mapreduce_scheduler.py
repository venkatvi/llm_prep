"""
Level 2 MapReduce Job Scheduler

A complete implementation of a sequential MapReduce scheduler that handles:
- Map → Shuffle → Reduce phases execution
- Intermediate file management between phases
- Task coordination and dependency management

This is a toy implementation for learning Level 2 concepts.
"""

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Callable, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import logging

# Import streaming capabilities
from streaming_processor import (
    StreamingConfig, StreamingMapTask, StreamingIntegration,
    LargeDatasetGenerator
)

# Import error handling capabilities
from error_handling import (
    ErrorRecoverySystem, RetryConfig, CheckpointConfig, FailureConfig,
    TaskCheckpoint, JobCheckpoint, RetryStrategy, FailureType,
    create_development_config, create_production_config, create_testing_config
)


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TaskInfo:
    """Information about a map or reduce task
    # Task ID, Task Type
    # Input and Output Files /Dirs
    # Status
    # Start and End Time
    # Retry Count
    # Logs / Error messages
    """
    task_id: str
    task_type: str  # 'map' or 'reduce'
    input_files: List[str]
    output_file: str
    status: str = 'pending'  # pending, running, completed, failed
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    retry_count: int = 0
    error_message: Optional[str] = None
    processing_strategy: str = 'traditional'  # 'traditional', 'split', 'stream'


@dataclass
class JobConfig:
    """Configuration for a MapReduce job
    # Job which takes on a task
    - Map function
    - Redeuce function
    - Number of reducers
    - Local combiner enabled ?
    - Local combiner definition
    - Intermediate Dirs
    - Input and output files / Dirs

    """
    job_name: str
    map_function: Callable
    reduce_function: Callable
    input_files: List[str]
    output_dir: str
    num_reduce_tasks: int = 4
    intermediate_dir: str = None
    max_retries: int = 3
    enable_combiner: bool = False
    combiner_function: Optional[Callable] = None

    # Streaming configuration
    enable_streaming: bool = False
    streaming_config: Optional[StreamingConfig] = None
    streaming_threshold_mb: int = 100  # Auto-enable streaming for files > 100MB
    num_map_tasks: Optional[int] = None  # Enable map task partitioning

    # Error handling configuration
    enable_error_recovery: bool = True
    retry_config: Optional[RetryConfig] = None
    checkpoint_config: Optional[CheckpointConfig] = None
    failure_config: Optional[FailureConfig] = None


class IntermediateFileManager:
    """Manages intermediate files between map and reduce phases
    Handler to create input, output, shuffle and intermediate directories 
    Handler to return IO files mapping to a given task id 
    
    """

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for different phases
        self.map_output_dir = self.base_dir / "map_output"
        self.shuffle_dir = self.base_dir / "shuffle"
        self.reduce_input_dir = self.base_dir / "reduce_input"

        for dir_path in [self.map_output_dir, self.shuffle_dir, self.reduce_input_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def get_map_output_file(self, map_task_id: str) -> str:
        """Get the output file path for a map task"""
        return str(self.map_output_dir / f"map_{map_task_id}.json")

    def get_reduce_input_file(self, reduce_task_id: str) -> str:
        """Get the input file path for a reduce task"""
        return str(self.reduce_input_dir / f"reduce_input_{reduce_task_id}.json")

    def get_shuffle_temp_file(self, partition_id: int) -> str:
        """Get temporary file for shuffle phase"""
        return str(self.shuffle_dir / f"shuffle_partition_{partition_id}.json")

    def cleanup(self):
        """Clean up all intermediate files"""
        if self.base_dir.exists():
            shutil.rmtree(self.base_dir)
            logger.info(f"Cleaned up intermediate directory: {self.base_dir}")


class TaskCoordinator:
    """Coordinates task execution and dependency management
    You can add as many tasks as you want to a coordinator 
    [task id] --> TaskInfo 
    [task id] --> List of prerequisite tasks ids to finish before executing this task.
    Completed set of tasks, failde set of tasks 
    """

    def __init__(self):
        self.tasks: Dict[str, TaskInfo] = {}
        self.dependencies: Dict[str, List[str]] = {}  # task_id -> list of prerequisite task_ids
        self.completed_tasks: set = set()
        self.failed_tasks: set = set()

    def add_task(self, task: TaskInfo, dependencies: List[str] = None):
        """Add a task with optional dependencies"""
        self.tasks[task.task_id] = task
        self.dependencies[task.task_id] = dependencies or []
        logger.info(f"Added task {task.task_id} with dependencies: {dependencies}")

    def get_ready_tasks(self) -> List[TaskInfo]:
        """Get tasks that are ready to run (all dependencies completed)"""
        ready_tasks = []

        for task_id, task in self.tasks.items():
            if (task.status == 'pending' and
                all(dep_id in self.completed_tasks for dep_id in self.dependencies[task_id])):
                ready_tasks.append(task)

        return ready_tasks

    def mark_task_completed(self, task_id: str):
        """Mark a task as completed"""
        if task_id in self.tasks:
            self.tasks[task_id].status = 'completed'
            self.tasks[task_id].end_time = time.time()
            self.completed_tasks.add(task_id)
            logger.info(f"Task {task_id} marked as completed")

    def mark_task_failed(self, task_id: str, error_message: str):
        """Mark a task as failed"""
        if task_id in self.tasks:
            self.tasks[task_id].status = 'failed'
            self.tasks[task_id].error_message = error_message
            self.tasks[task_id].end_time = time.time()
            self.failed_tasks.add(task_id)
            logger.error(f"Task {task_id} marked as failed: {error_message}")

    def all_tasks_completed(self) -> bool:
        """Check if all tasks are completed"""
        return len(self.completed_tasks) == len(self.tasks)

    def has_failed_tasks(self) -> bool:
        """Check if any tasks have failed"""
        return len(self.failed_tasks) > 0

    def get_task_stats(self) -> Dict[str, int]:
        """Get statistics about task execution
        number of completed, failed and pending tasks 
        """
        stats = defaultdict(int)
        for task in self.tasks.values():
            stats[task.status] += 1
        return dict(stats)


class MapReduceScheduler:
    """Main MapReduce job scheduler with streaming and error recovery support"""

    def __init__(self,
                 max_concurrent_tasks: int = 4,
                 recovery_system: ErrorRecoverySystem = None):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.coordinator = TaskCoordinator()
        self.file_manager: Optional[IntermediateFileManager] = None
        self.streaming_tasks: Dict[str, StreamingMapTask] = {}  # Cache streaming tasks
        self.recovery_system = recovery_system or ErrorRecoverySystem(*create_development_config())
        self.job_checkpoints: Dict[str, JobCheckpoint] = {}
        self.current_job_id: Optional[str] = None

    def execute_job(self, job_config: JobConfig) -> bool:
        """Execute a complete MapReduce job with streaming and error recovery support"""
        logger.info(f"Starting MapReduce job: {job_config.job_name}")

        # Generate unique job ID for checkpointing
        self.current_job_id = f"{job_config.job_name}_{int(time.time())}"

        # Initialize error recovery system with job config
        if job_config.enable_error_recovery:
            self._initialize_error_recovery(job_config)

        # Attempt to recover from checkpoint first
        if job_config.enable_error_recovery:
            recovered_checkpoint = self.recovery_system.recover_job_from_checkpoint(
                job_config.job_name, self.current_job_id
            )
            if recovered_checkpoint:
                logger.info(f"Resuming job from checkpoint with {len(recovered_checkpoint.completed_tasks)} completed tasks")
                return self._resume_from_checkpoint(job_config, recovered_checkpoint)

        try:
            # Initialize streaming configuration if not provided
            if job_config.enable_streaming and not job_config.streaming_config:
                job_config.streaming_config = StreamingIntegration.create_streaming_config()

            # Auto-enable streaming for large files
            if not job_config.enable_streaming:
                for input_file in job_config.input_files:
                    if StreamingIntegration.should_use_streaming(input_file, job_config.streaming_threshold_mb):
                        logger.info(f"Auto-enabling streaming for large file: {input_file}")
                        job_config.enable_streaming = True
                        job_config.streaming_config = StreamingIntegration.create_streaming_config()
                        break

            # Initialize intermediate file manager
            intermediate_dir = job_config.intermediate_dir or tempfile.mkdtemp(prefix="mapreduce_")
            self.file_manager = IntermediateFileManager(intermediate_dir)

            # Create output directory
            output_path = Path(job_config.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Prepare input files using hybrid approach
            prepared_files = self._prepare_input_files(job_config)

            # Phase 1: Create and schedule map tasks
            map_task_ids = self._create_map_tasks(job_config, prepared_files)

            # Phase 2: Create shuffle dependency (virtual task)
            shuffle_task_id = self._create_shuffle_task(map_task_ids, job_config.num_reduce_tasks)

            # Phase 3: Create and schedule reduce tasks
            reduce_task_ids = self._create_reduce_tasks(job_config, shuffle_task_id)

            # Execute all tasks with dependency management
            success = self._execute_all_tasks(job_config)

            if success:
                # Merge final outputs
                self._merge_final_outputs(job_config, reduce_task_ids)
                logger.info(f"MapReduce job {job_config.job_name} completed successfully")
            else:
                logger.error(f"MapReduce job {job_config.job_name} failed")

            return success

        except Exception as e:
            logger.error(f"Error executing job {job_config.job_name}: {str(e)}")
            return False

        finally:
            # Cleanup intermediate files
            if self.file_manager:
                self.file_manager.cleanup()

    def _initialize_error_recovery(self, job_config: JobConfig):
        """Initialize error recovery system with job-specific configuration"""
        if job_config.retry_config:
            self.recovery_system.retry_manager.config = job_config.retry_config

        if job_config.checkpoint_config:
            self.recovery_system.checkpoint_manager.config = job_config.checkpoint_config

        if job_config.failure_config:
            self.recovery_system.failure_simulator.config = job_config.failure_config

    def _create_job_checkpoint(self, job_config: JobConfig, completed_task_id: str, status: str):
        """Create a checkpoint for the current job state"""
        if not job_config.enable_error_recovery or not self.current_job_id:
            return

        # Collect task checkpoints
        task_checkpoints = {}
        for task_id, task in self.coordinator.tasks.items():
            checkpoint = self.recovery_system.checkpoint_manager.create_task_checkpoint(
                task_id=task.task_id,
                task_type=task.task_type,
                status=task.status,
                progress=1.0 if task.status == 'completed' else 0.0,
                input_files=task.input_files,
                output_file=task.output_file
            )
            task_checkpoints[task_id] = checkpoint

        # Create dependency graph
        dependency_graph = {}
        for task_id, deps in self.coordinator.dependencies.items():
            dependency_graph[task_id] = list(deps)

        # Save job checkpoint
        self.recovery_system.create_job_checkpoint(
            job_name=job_config.job_name,
            job_id=self.current_job_id,
            status=status,
            task_checkpoints=task_checkpoints,
            dependency_graph=dependency_graph,
            completed_tasks=list(self.coordinator.completed_tasks),
            failed_tasks=list(self.coordinator.failed_tasks)
        )

    def _resume_from_checkpoint(self, job_config: JobConfig, checkpoint: JobCheckpoint) -> bool:
        """Resume job execution from a checkpoint"""
        logger.info(f"Resuming job {job_config.job_name} from checkpoint")

        # Restore task states
        for task_id, task_checkpoint in checkpoint.task_checkpoints.items():
            if task_checkpoint.status == 'completed':
                # Mark task as completed in coordinator
                if task_id in self.coordinator.tasks:
                    self.coordinator.mark_task_completed(task_id)

        # Continue execution from where we left off
        return self._execute_all_tasks(job_config)

    def get_error_recovery_stats(self) -> Dict[str, int]:
        """Get error recovery statistics"""
        return self.recovery_system.get_recovery_statistics()

    def enable_failure_simulation(self, failure_rate: float = 0.1,
                                failure_types: List[FailureType] = None,
                                target_tasks: List[str] = None):
        """Enable failure simulation for testing"""
        self.recovery_system.failure_simulator.config.enabled = True
        self.recovery_system.failure_simulator.config.failure_rate = failure_rate
        if failure_types:
            self.recovery_system.failure_simulator.config.failure_types = failure_types
        if target_tasks:
            self.recovery_system.failure_simulator.config.target_tasks = target_tasks

    def disable_failure_simulation(self):
        """Disable failure simulation"""
        self.recovery_system.failure_simulator.config.enabled = False

    def _should_split_vs_stream(self, file_path: str, job_config: JobConfig) -> str:
        """
        Determine the optimal processing strategy for a file based on size and configuration.

        Returns:
            "traditional": Use standard file reading (small files)
            "split": Split file into chunks for parallel traditional processing
            "stream": Use streaming for memory-efficient processing
        """
        try:
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        except OSError:
            logger.warning(f"Could not get size for {file_path}, defaulting to traditional")
            return "traditional"

        # Decision matrix based on file size and configuration
        if file_size_mb < 50:  # Small files
            return "traditional"

        elif file_size_mb < 300 and job_config.num_map_tasks and job_config.num_map_tasks > 1:
            # Medium files: split for parallelism if multiple map tasks requested
            return "split"

        elif file_size_mb >= job_config.streaming_threshold_mb:
            # Large files: use streaming for memory efficiency
            return "stream"

        else:
            # Default to traditional for edge cases
            return "traditional"

    def _prepare_input_files(self, job_config: JobConfig) -> List[Tuple[str, str]]:
        """
        Prepare input files using hybrid approach: traditional, split, or stream.

        Returns:
            List of (file_path, processing_strategy) tuples
        """
        prepared_files = []
        split_dir = Path(self.file_manager.base_dir) / "split_inputs"

        for input_file in job_config.input_files:
            strategy = self._should_split_vs_stream(input_file, job_config)

            if strategy == "split":
                logger.info(f"Splitting file {input_file} for parallel processing ({job_config.num_map_tasks} tasks)")

                # Create split directory
                split_dir.mkdir(parents=True, exist_ok=True)

                # Split file into chunks
                split_files = StreamingIntegration.split_large_file_for_map_tasks(
                    input_file=input_file,
                    num_map_tasks=job_config.num_map_tasks,
                    output_dir=str(split_dir),
                    config=job_config.streaming_config or StreamingIntegration.create_streaming_config()
                )

                # Each split file will use traditional processing
                for split_file in split_files:
                    prepared_files.append((split_file, "traditional"))

            else:
                # Use file as-is with the determined strategy
                prepared_files.append((input_file, strategy))

                file_size_mb = os.path.getsize(input_file) / (1024 * 1024)
                logger.info(f"File {input_file} ({file_size_mb:.1f}MB) will use {strategy} processing")

        return prepared_files

    def _create_map_tasks(self, job_config: JobConfig, prepared_files: List[Tuple[str, str]]) -> List[str]:
        """Create map tasks for prepared input files with their processing strategies"""
        map_task_ids = []

        for i, (input_file, strategy) in enumerate(prepared_files):
            task_id = f"map_{i}"
            output_file = self.file_manager.get_map_output_file(task_id)

            task = TaskInfo(
                task_id=task_id,
                task_type='map',
                input_files=[input_file],
                output_file=output_file
            )

            self.coordinator.add_task(task)
            map_task_ids.append(task_id)

            # Store processing strategy for this task
            task.processing_strategy = strategy

            # Initialize streaming task only if strategy is "stream"
            if strategy == "stream":
                streaming_config = job_config.streaming_config or StreamingIntegration.create_streaming_config()
                streaming_task = StreamingMapTask(streaming_config)
                self.streaming_tasks[task_id] = streaming_task

        # Count strategies for logging
        strategy_counts = {}
        for _, strategy in prepared_files:
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        strategy_summary = ", ".join([f"{strategy}: {count}" for strategy, count in strategy_counts.items()])
        logger.info(f"Created {len(map_task_ids)} map tasks ({strategy_summary})")
        return map_task_ids

    def _create_shuffle_task(self, map_task_ids: List[str], num_reduce_tasks: int) -> str:
        """Create a shuffle task that depends on all map tasks"""
        shuffle_task_id = "shuffle"

        # Create a virtual shuffle task that processes all map outputs
        task = TaskInfo(
            task_id=shuffle_task_id,
            task_type='shuffle',
            input_files=[],  # Will be populated from map outputs
            output_file=""   # Shuffle creates multiple output files
        )

        self.coordinator.add_task(task, dependencies=map_task_ids)
        logger.info(f"Created shuffle task depending on {len(map_task_ids)} map tasks")
        return shuffle_task_id

    def _create_reduce_tasks(self, job_config: JobConfig, shuffle_task_id: str) -> List[str]:
        """Create reduce tasks that depend on shuffle completion"""
        reduce_task_ids = []

        for i in range(job_config.num_reduce_tasks):
            task_id = f"reduce_{i}"
            input_file = self.file_manager.get_reduce_input_file(task_id)
            output_file = str(Path(job_config.output_dir) / f"part-{i:05d}")

            task = TaskInfo(
                task_id=task_id,
                task_type='reduce',
                input_files=[input_file],
                output_file=output_file
            )

            self.coordinator.add_task(task, dependencies=[shuffle_task_id])
            reduce_task_ids.append(task_id)

        logger.info(f"Created {len(reduce_task_ids)} reduce tasks")
        return reduce_task_ids

    def _execute_all_tasks(self, job_config: JobConfig) -> bool:
        """Execute all tasks respecting dependencies"""
        with ThreadPoolExecutor(max_workers=self.max_concurrent_tasks) as executor:
            running_tasks = {}

            while not self.coordinator.all_tasks_completed() and not self.coordinator.has_failed_tasks():
                # Get tasks ready to run
                ready_tasks = self.coordinator.get_ready_tasks()

                # Submit ready tasks that aren't already running
                for task in ready_tasks:
                    if task.task_id not in running_tasks:
                        task.status = 'running'
                        task.start_time = time.time()

                        future = executor.submit(self._execute_task, task, job_config)
                        running_tasks[task.task_id] = future
                        logger.info(f"Started task {task.task_id}")

                # Check for completed tasks
                completed_futures = []
                for task_id, future in running_tasks.items():
                    if future.done():
                        try:
                            success = future.result()
                            if success:
                                self.coordinator.mark_task_completed(task_id)
                            else:
                                self.coordinator.mark_task_failed(task_id, "Task execution failed")
                        except Exception as e:
                            self.coordinator.mark_task_failed(task_id, str(e))

                        completed_futures.append(task_id)

                # Remove completed futures
                for task_id in completed_futures:
                    del running_tasks[task_id]

                # Brief pause to avoid busy waiting
                if not ready_tasks and running_tasks:
                    time.sleep(0.1)

                # Print progress
                stats = self.coordinator.get_task_stats()
                total_tasks = len(self.coordinator.tasks)
                logger.info(f"Task progress: {stats.get('completed', 0)}/{total_tasks} completed, "
                           f"{stats.get('running', 0)} running, {stats.get('failed', 0)} failed")

        return self.coordinator.all_tasks_completed() and not self.coordinator.has_failed_tasks()

    def _execute_task(self, task: TaskInfo, job_config: JobConfig) -> bool:
        """Execute a single task with error recovery support"""

        # Create task execution function
        def task_function():
            if task.task_type == 'map':
                return self._execute_map_task(task, job_config)
            elif task.task_type == 'shuffle':
                return self._execute_shuffle_task(task, job_config)
            elif task.task_type == 'reduce':
                return self._execute_reduce_task(task, job_config)
            else:
                logger.error(f"Unknown task type: {task.task_type}")
                return False

        # Execute with error recovery if enabled
        if job_config.enable_error_recovery:
            success, result, checkpoint = self.recovery_system.execute_task_with_recovery(
                task_id=task.task_id,
                task_type=task.task_type,
                task_function=task_function,
                input_files=task.input_files,
                output_file=task.output_file
            )

            # Update task info with retry information
            task.retry_count = checkpoint.retry_count
            if checkpoint.error_history:
                task.error_message = "; ".join(checkpoint.error_history)

            # Create job checkpoint after each task
            if success:
                self._create_job_checkpoint(job_config, task.task_id, "completed")

            return success

        else:
            # Execute without error recovery (legacy mode)
            try:
                return task_function()
            except Exception as e:
                logger.error(f"Error executing task {task.task_id}: {str(e)}")
                task.error_message = str(e)
                return False

    def _execute_map_task(self, task: TaskInfo, job_config: JobConfig) -> bool:
        """Execute a map task using the hybrid approach based on processing strategy"""
        logger.info(f"Executing map task {task.task_id} using {task.processing_strategy} strategy")

        # Choose execution method based on processing strategy
        if task.processing_strategy == "stream":
            return self._execute_streaming_map_task(task, job_config)
        else:
            # Both "traditional" and "split" use traditional processing
            # (split files are already small enough for traditional processing)
            return self._execute_traditional_map_task(task, job_config)

    def _execute_streaming_map_task(self, task: TaskInfo, job_config: JobConfig) -> bool:
        """Execute map task using streaming processor"""
        streaming_task = self.streaming_tasks[task.task_id]

        for input_file in task.input_files:
            success = streaming_task.execute_streaming_map(
                input_file=input_file,
                output_file=task.output_file,
                map_function=job_config.map_function,
                num_reduce_tasks=job_config.num_reduce_tasks
            )

            if not success:
                return False

        return True

    def _execute_traditional_map_task(self, task: TaskInfo, job_config: JobConfig) -> bool:
        """Execute map task using traditional file reading"""
        # Read input file and apply map function
        intermediate_data = []

        for input_file in task.input_files:
            with open(input_file, 'r') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if line:
                        try:
                            # Apply map function
                            map_results = job_config.map_function(line)

                            # Store intermediate key-value pairs
                            for key, value in map_results:
                                # Add partition information for shuffle phase
                                partition_id = self._hash_partition(key, job_config.num_reduce_tasks)
                                intermediate_data.append({
                                    'key': key,
                                    'value': value,
                                    'partition': partition_id
                                })

                        except Exception as e:
                            logger.error(f"Error processing line {line_num} in {input_file}: {str(e)}")
                            return False

        # Write intermediate data to output file
        with open(task.output_file, 'w') as f:
            for record in intermediate_data:
                f.write(json.dumps(record) + '\n')

        logger.info(f"Map task {task.task_id} completed, wrote {len(intermediate_data)} records")
        return True

    def _execute_shuffle_task(self, task: TaskInfo, job_config: JobConfig) -> bool:
        """Execute shuffle phase - sort and partition intermediate data"""
        logger.info(f"Executing shuffle task {task.task_id}")

        # Collect all intermediate data from map tasks
        partitioned_data = defaultdict(list)

        # Read all map output files
        for map_task_id in range(len(job_config.input_files)):
            map_output_file = self.file_manager.get_map_output_file(f"map_{map_task_id}")

            with open(map_output_file, 'r') as f:
                for line in f:
                    record = json.loads(line.strip())
                    partition_id = record['partition']
                    partitioned_data[partition_id].append((record['key'], record['value']))

        # Sort data within each partition and write to reduce input files
        # Shuffle per partition is happening here! 
        for partition_id in range(job_config.num_reduce_tasks):
            reduce_input_file = self.file_manager.get_reduce_input_file(f"reduce_{partition_id}")

            # Sort by key for efficient reduce processing
            partition_data = sorted(partitioned_data[partition_id], key=lambda x: x[0])

            with open(reduce_input_file, 'w') as f:
                for key, value in partition_data:
                    f.write(json.dumps({'key': key, 'value': value}) + '\n')

            logger.info(f"Wrote {len(partition_data)} records to partition {partition_id}")

        logger.info(f"Shuffle task {task.task_id} completed")
        return True

    def _execute_reduce_task(self, task: TaskInfo, job_config: JobConfig) -> bool:
        """Execute a reduce task"""
        logger.info(f"Executing reduce task {task.task_id}")

        # Group values by key
        grouped_data = defaultdict(list)

        # Read sorted input data
        for input_file in task.input_files:
            with open(input_file, 'r') as f:
                for line in f:
                    record = json.loads(line.strip())
                    key = record['key']
                    value = record['value']
                    grouped_data[key].append(value)

        # Apply reduce function to each key group
        final_results = []
        for key, values in grouped_data.items():
            try:
                # Apply reduce function
                reduced_value = job_config.reduce_function(key, values)
                final_results.append((key, reduced_value))

            except Exception as e:
                logger.error(f"Error reducing key {key}: {str(e)}")
                return False

        # Write final results
        with open(task.output_file, 'w') as f:
            for key, value in sorted(final_results):
                f.write(f"{key}\t{value}\n")

        logger.info(f"Reduce task {task.task_id} completed, wrote {len(final_results)} records")
        return True

    def _hash_partition(self, key: str, num_partitions: int) -> int:
        """Hash partitioning function"""
        return hash(key) % num_partitions

    def _merge_final_outputs(self, job_config: JobConfig, reduce_task_ids: List[str]):
        """Merge all reduce outputs into a single final output"""
        final_output_file = Path(job_config.output_dir) / "final_output.txt"

        with open(final_output_file, 'w') as outf:
            for task_id in sorted(reduce_task_ids):
                task_num = int(task_id.split('_')[1])
                reduce_output_file = Path(job_config.output_dir) / f"part-{task_num:05d}"

                if reduce_output_file.exists():
                    with open(reduce_output_file, 'r') as inf:
                        outf.write(inf.read())

        logger.info(f"Final output written to {final_output_file}")


# Example Word Count Functions
def word_count_map(line: str) -> List[Tuple[str, int]]:
    """Map function for word count"""
    words = line.lower().strip().split()
    return [(word, 1) for word in words if word.isalpha()]


def word_count_reduce(key: str, values: List[int]) -> int:
    """Reduce function for word count"""
    return sum(values)


def create_sample_data(output_dir: str, num_files: int = 3, lines_per_file: int = 1000):
    """Create sample text data for testing"""
    import random

    sample_words = [
        'the', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog',
        'hello', 'world', 'mapreduce', 'is', 'awesome', 'distributed', 'computing',
        'big', 'data', 'processing', 'scalable', 'systems'
    ]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    input_files = []

    for i in range(num_files):
        file_path = output_path / f"input_{i}.txt"
        input_files.append(str(file_path))

        with open(file_path, 'w') as f:
            for _ in range(lines_per_file):
                # Generate random sentences
                sentence_length = random.randint(5, 15)
                words = random.choices(sample_words, k=sentence_length)
                f.write(' '.join(words) + '\n')

    logger.info(f"Created {num_files} sample files with {lines_per_file} lines each")
    return input_files


if __name__ == "__main__":
    # Example usage of the MapReduce scheduler

    # Create sample data
    data_dir = "/tmp/mapreduce_test_data"
    input_files = create_sample_data(data_dir, num_files=3, lines_per_file=500)

    # Configure the job
    job_config = JobConfig(
        job_name="word_count_example",
        map_function=word_count_map,
        reduce_function=word_count_reduce,
        input_files=input_files,
        output_dir="/tmp/mapreduce_output",
        num_reduce_tasks=4,
        max_retries=3
    )

    # Create and run the scheduler
    scheduler = MapReduceScheduler(max_concurrent_tasks=2)
    success = scheduler.execute_job(job_config)

    if success:
        print("✅ MapReduce job completed successfully!")
        print("Check output in /tmp/mapreduce_output/final_output.txt")
    else:
        print("❌ MapReduce job failed!")