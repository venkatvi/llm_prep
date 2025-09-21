"""
Error Handling and Recovery System for MapReduce

This module implements comprehensive error handling, retry logic,
checkpointing, and failure recovery mechanisms for the MapReduce scheduler.

Key Features:
1. Configurable retry strategies with exponential backoff
2. Checkpointing for long-running tasks and job state persistence
3. Failure simulation and recovery testing
4. Task-level and job-level error recovery
5. Partial failure handling and graceful degradation
"""

import json
import os
import time
import random
import pickle
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from concurrent.futures import Future

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    """Different retry strategies for failed tasks"""
    IMMEDIATE = "immediate"
    LINEAR_BACKOFF = "linear_backoff"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    FIXED_DELAY = "fixed_delay"


class FailureType(Enum):
    """Types of failures that can be simulated"""
    TASK_CRASH = "task_crash"
    NETWORK_FAILURE = "network_failure"
    DISK_FULL = "disk_full"
    MEMORY_ERROR = "memory_error"
    TIMEOUT = "timeout"
    CORRUPTION = "corruption"
    RANDOM_FAILURE = "random_failure"


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_retries: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True  # Add randomness to prevent thundering herd


@dataclass
class CheckpointConfig:
    """Configuration for checkpointing behavior"""
    enabled: bool = True
    checkpoint_dir: str = "/tmp/mapreduce_checkpoints"
    checkpoint_interval_seconds: int = 30
    max_checkpoints_to_keep: int = 5
    auto_cleanup: bool = True


@dataclass
class FailureConfig:
    """Configuration for failure simulation"""
    enabled: bool = False
    failure_rate: float = 0.1  # 10% chance of failure
    failure_types: List[FailureType] = None
    target_tasks: List[str] = None  # Specific tasks to fail, None for random


@dataclass
class TaskCheckpoint:
    """Checkpoint data for a single task"""
    task_id: str
    task_type: str
    status: str
    progress: float
    input_files: List[str]
    output_file: str
    partial_results: Optional[Dict] = None
    timestamp: float = None
    retry_count: int = 0
    error_history: List[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.error_history is None:
            self.error_history = []


@dataclass
class JobCheckpoint:
    """Checkpoint data for entire job"""
    job_name: str
    job_id: str
    status: str
    task_checkpoints: Dict[str, TaskCheckpoint]
    dependency_graph: Dict[str, List[str]]
    completed_tasks: List[str]
    failed_tasks: List[str]
    timestamp: float = None
    total_progress: float = 0.0

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class RetryManager:
    """Manages retry logic for failed tasks"""

    def __init__(self, config: RetryConfig):
        self.config = config

    def should_retry(self, task_id: str, retry_count: int, error: Exception) -> bool:
        """Determine if a task should be retried based on failure count and type"""
        if retry_count >= self.config.max_retries:
            logger.warning(f"Task {task_id} exceeded max retries ({self.config.max_retries})")
            return False

        # Don't retry certain types of errors
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            logger.info(f"Task {task_id} failed with non-retryable error: {type(error).__name__}")
            return False

        return True

    def calculate_delay(self, retry_count: int) -> float:
        """Calculate delay before next retry attempt"""
        if self.config.strategy == RetryStrategy.IMMEDIATE:
            delay = 0
        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.base_delay_seconds * retry_count
        elif self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.base_delay_seconds * (self.config.backoff_multiplier ** retry_count)
        elif self.config.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.config.base_delay_seconds
        else:
            delay = self.config.base_delay_seconds

        # Cap at max delay
        delay = min(delay, self.config.max_delay_seconds)

        # Add jitter to prevent thundering herd
        if self.config.jitter:
            jitter_factor = random.uniform(0.5, 1.5)
            delay *= jitter_factor

        return delay

    def execute_with_retry(self,
                          task_id: str,
                          task_function: Callable,
                          *args, **kwargs) -> Tuple[bool, Any, List[str]]:
        """Execute a task with retry logic"""
        retry_count = 0
        errors = []

        while retry_count <= self.config.max_retries:
            try:
                logger.info(f"Executing task {task_id} (attempt {retry_count + 1})")
                result = task_function(*args, **kwargs)
                if retry_count > 0:
                    logger.info(f"Task {task_id} succeeded after {retry_count} retries")
                return True, result, errors

            except Exception as e:
                retry_count += 1
                error_msg = f"Attempt {retry_count}: {str(e)}"
                errors.append(error_msg)
                logger.error(f"Task {task_id} failed: {error_msg}")

                if not self.should_retry(task_id, retry_count, e):
                    break

                if retry_count <= self.config.max_retries:
                    delay = self.calculate_delay(retry_count)
                    logger.info(f"Retrying task {task_id} after {delay:.2f} seconds")
                    time.sleep(delay)

        logger.error(f"Task {task_id} failed after {retry_count} attempts")
        return False, None, errors


class CheckpointManager:
    """Manages checkpointing for job and task state persistence"""

    def __init__(self, config: CheckpointConfig):
        self.config = config
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.active_checkpoints: Dict[str, JobCheckpoint] = {}

    def save_job_checkpoint(self, job_checkpoint: JobCheckpoint) -> str:
        """Save job checkpoint to disk"""
        if not self.config.enabled:
            return ""

        checkpoint_file = self.checkpoint_dir / f"{job_checkpoint.job_name}_{job_checkpoint.job_id}.checkpoint"

        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(job_checkpoint, f)

            self.active_checkpoints[job_checkpoint.job_id] = job_checkpoint
            logger.info(f"Saved checkpoint for job {job_checkpoint.job_name} to {checkpoint_file}")

            # Cleanup old checkpoints
            if self.config.auto_cleanup:
                self._cleanup_old_checkpoints(job_checkpoint.job_name)

            return str(checkpoint_file)

        except Exception as e:
            logger.error(f"Failed to save checkpoint for job {job_checkpoint.job_name}: {e}")
            return ""

    def load_job_checkpoint(self, job_name: str, job_id: str) -> Optional[JobCheckpoint]:
        """Load job checkpoint from disk"""
        if not self.config.enabled:
            return None

        checkpoint_file = self.checkpoint_dir / f"{job_name}_{job_id}.checkpoint"

        if not checkpoint_file.exists():
            return None

        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)

            logger.info(f"Loaded checkpoint for job {job_name} from {checkpoint_file}")
            return checkpoint

        except Exception as e:
            logger.error(f"Failed to load checkpoint for job {job_name}: {e}")
            return None

    def list_available_checkpoints(self, job_name: str = None) -> List[str]:
        """List available checkpoint files"""
        pattern = f"{job_name}_*.checkpoint" if job_name else "*.checkpoint"
        return [str(f) for f in self.checkpoint_dir.glob(pattern)]

    def create_task_checkpoint(self,
                              task_id: str,
                              task_type: str,
                              status: str,
                              progress: float,
                              input_files: List[str],
                              output_file: str,
                              partial_results: Dict = None) -> TaskCheckpoint:
        """Create a checkpoint for a single task"""
        return TaskCheckpoint(
            task_id=task_id,
            task_type=task_type,
            status=status,
            progress=progress,
            input_files=input_files,
            output_file=output_file,
            partial_results=partial_results or {}
        )

    def _cleanup_old_checkpoints(self, job_name: str):
        """Remove old checkpoint files to save disk space"""
        pattern = f"{job_name}_*.checkpoint"
        checkpoint_files = sorted(
            self.checkpoint_dir.glob(pattern),
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )

        # Keep only the most recent N checkpoints
        for old_checkpoint in checkpoint_files[self.config.max_checkpoints_to_keep:]:
            try:
                old_checkpoint.unlink()
                logger.info(f"Cleaned up old checkpoint: {old_checkpoint}")
            except Exception as e:
                logger.warning(f"Failed to cleanup checkpoint {old_checkpoint}: {e}")


class FailureSimulator:
    """Simulates various types of failures for testing recovery mechanisms"""

    def __init__(self, config: FailureConfig):
        self.config = config
        if config.failure_types is None:
            self.config.failure_types = list(FailureType)

    def should_fail(self, task_id: str) -> Tuple[bool, FailureType]:
        """Determine if a task should fail and what type of failure"""
        if not self.config.enabled:
            return False, None

        # Check if this specific task is targeted
        if self.config.target_tasks and task_id not in self.config.target_tasks:
            return False, None

        # Random failure based on failure rate
        if random.random() < self.config.failure_rate:
            failure_type = random.choice(self.config.failure_types)
            return True, failure_type

        return False, None

    def simulate_failure(self, task_id: str, failure_type: FailureType):
        """Simulate a specific type of failure"""
        logger.warning(f"Simulating {failure_type.value} for task {task_id}")

        if failure_type == FailureType.TASK_CRASH:
            raise RuntimeError(f"Simulated task crash for {task_id}")

        elif failure_type == FailureType.NETWORK_FAILURE:
            raise ConnectionError(f"Simulated network failure for {task_id}")

        elif failure_type == FailureType.DISK_FULL:
            raise OSError(f"Simulated disk full error for {task_id}")

        elif failure_type == FailureType.MEMORY_ERROR:
            raise MemoryError(f"Simulated memory error for {task_id}")

        elif failure_type == FailureType.TIMEOUT:
            raise TimeoutError(f"Simulated timeout for {task_id}")

        elif failure_type == FailureType.CORRUPTION:
            raise ValueError(f"Simulated data corruption for {task_id}")

        elif failure_type == FailureType.RANDOM_FAILURE:
            failure_types = [f for f in FailureType if f != FailureType.RANDOM_FAILURE]
            random_failure = random.choice(failure_types)
            self.simulate_failure(task_id, random_failure)


class ErrorRecoverySystem:
    """Central system that coordinates error handling, retries, and checkpointing"""

    def __init__(self,
                 retry_config: RetryConfig = None,
                 checkpoint_config: CheckpointConfig = None,
                 failure_config: FailureConfig = None):

        self.retry_manager = RetryManager(retry_config or RetryConfig())
        self.checkpoint_manager = CheckpointManager(checkpoint_config or CheckpointConfig())
        self.failure_simulator = FailureSimulator(failure_config or FailureConfig())

        self.recovery_stats = {
            'total_tasks': 0,
            'failed_tasks': 0,
            'recovered_tasks': 0,
            'permanently_failed_tasks': 0,
            'checkpoints_created': 0,
            'checkpoints_restored': 0
        }

    def execute_task_with_recovery(self,
                                  task_id: str,
                                  task_type: str,
                                  task_function: Callable,
                                  input_files: List[str],
                                  output_file: str,
                                  *args, **kwargs) -> Tuple[bool, Any, TaskCheckpoint]:
        """Execute a task with full error recovery support"""

        self.recovery_stats['total_tasks'] += 1

        # Check for failure simulation
        should_fail, failure_type = self.failure_simulator.should_fail(task_id)

        # Create initial checkpoint
        checkpoint = self.checkpoint_manager.create_task_checkpoint(
            task_id=task_id,
            task_type=task_type,
            status='running',
            progress=0.0,
            input_files=input_files,
            output_file=output_file
        )

        def wrapped_task_function(*args, **kwargs):
            # Simulate failure if needed
            if should_fail:
                self.failure_simulator.simulate_failure(task_id, failure_type)

            # Execute the actual task
            return task_function(*args, **kwargs)

        # Execute with retry logic
        success, result, errors = self.retry_manager.execute_with_retry(
            task_id, wrapped_task_function, *args, **kwargs
        )

        # Update checkpoint based on result
        if success:
            checkpoint.status = 'completed'
            checkpoint.progress = 1.0
        else:
            checkpoint.status = 'failed'
            checkpoint.error_history.extend(errors)
            self.recovery_stats['failed_tasks'] += 1

            if checkpoint.retry_count > 0:
                self.recovery_stats['permanently_failed_tasks'] += 1
            else:
                self.recovery_stats['recovered_tasks'] += 1

        checkpoint.retry_count = len(errors)
        checkpoint.timestamp = time.time()

        return success, result, checkpoint

    def create_job_checkpoint(self,
                            job_name: str,
                            job_id: str,
                            status: str,
                            task_checkpoints: Dict[str, TaskCheckpoint],
                            dependency_graph: Dict[str, List[str]],
                            completed_tasks: List[str],
                            failed_tasks: List[str]) -> str:
        """Create a checkpoint for the entire job"""

        total_progress = len(completed_tasks) / len(task_checkpoints) if task_checkpoints else 0.0

        job_checkpoint = JobCheckpoint(
            job_name=job_name,
            job_id=job_id,
            status=status,
            task_checkpoints=task_checkpoints,
            dependency_graph=dependency_graph,
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            total_progress=total_progress
        )

        checkpoint_file = self.checkpoint_manager.save_job_checkpoint(job_checkpoint)
        if checkpoint_file:
            self.recovery_stats['checkpoints_created'] += 1

        return checkpoint_file

    def recover_job_from_checkpoint(self, job_name: str, job_id: str) -> Optional[JobCheckpoint]:
        """Attempt to recover a job from its latest checkpoint"""

        checkpoint = self.checkpoint_manager.load_job_checkpoint(job_name, job_id)
        if checkpoint:
            self.recovery_stats['checkpoints_restored'] += 1
            logger.info(f"Recovered job {job_name} from checkpoint with {len(checkpoint.completed_tasks)} completed tasks")

        return checkpoint

    def get_recovery_statistics(self) -> Dict[str, int]:
        """Get statistics about error recovery performance"""
        return self.recovery_stats.copy()

    def reset_statistics(self):
        """Reset recovery statistics"""
        for key in self.recovery_stats:
            self.recovery_stats[key] = 0


# Convenience functions for common configurations
def create_development_config() -> Tuple[RetryConfig, CheckpointConfig, FailureConfig]:
    """Create configuration suitable for development and testing"""
    return (
        RetryConfig(max_retries=2, strategy=RetryStrategy.FIXED_DELAY, base_delay_seconds=0.5),
        CheckpointConfig(enabled=True, checkpoint_interval_seconds=10),
        FailureConfig(enabled=False)
    )


def create_production_config() -> Tuple[RetryConfig, CheckpointConfig, FailureConfig]:
    """Create configuration suitable for production use"""
    return (
        RetryConfig(max_retries=5, strategy=RetryStrategy.EXPONENTIAL_BACKOFF, base_delay_seconds=2.0),
        CheckpointConfig(enabled=True, checkpoint_interval_seconds=60, max_checkpoints_to_keep=10),
        FailureConfig(enabled=False)
    )


def create_testing_config() -> Tuple[RetryConfig, CheckpointConfig, FailureConfig]:
    """Create configuration suitable for failure testing"""
    return (
        RetryConfig(max_retries=3, strategy=RetryStrategy.LINEAR_BACKOFF, base_delay_seconds=0.1),
        CheckpointConfig(enabled=True, checkpoint_interval_seconds=5),
        FailureConfig(enabled=True, failure_rate=0.2, failure_types=[FailureType.TASK_CRASH, FailureType.TIMEOUT])
    )