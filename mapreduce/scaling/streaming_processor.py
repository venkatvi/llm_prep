"""
File Streaming Processor for Large Dataset Handling

This module implements streaming file processing capabilities for MapReduce tasks
that need to handle files larger than available RAM. It provides:

1. Chunked file reading with configurable buffer sizes
2. Line boundary preservation across chunk boundaries
3. Memory-efficient streaming for map tasks
4. Large dataset generation for testing

Key Features:
- Process files of any size with bounded memory usage
- Handle edge cases like records spanning chunk boundaries
- Configurable chunk sizes and buffer management
- Integration with existing MapReduce scheduler
"""

import os
import io
import json
import mmap
import tempfile
import random
import string
from pathlib import Path
from typing import Iterator, List, Optional, Callable, Tuple, Generator
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class StreamingConfig:
    """Configuration for streaming file processing"""
    chunk_size_mb: int = 64  # Size of each chunk in MB
    buffer_size_kb: int = 4  # Buffer size for line boundary handling
    max_line_size_kb: int = 1024  # Maximum expected line size
    enable_mmap: bool = True  # Use memory mapping for large files
    temp_dir: Optional[str] = None  # Directory for temporary files


class ChunkedFileReader:
    """
    Reads large files in chunks while preserving line boundaries

    This class handles the complex task of reading files in fixed-size chunks
    while ensuring that no lines are split across chunk boundaries.
    """

    def __init__(self, config: StreamingConfig):
        self.config = config
        self.chunk_size_bytes = config.chunk_size_mb * 1024 * 1024
        self.buffer_size_bytes = config.buffer_size_kb * 1024
        self.max_line_size_bytes = config.max_line_size_kb * 1024

    def read_chunks(self, file_path: str) -> Iterator[List[str]]:
        """
        Read file in chunks, yielding lists of complete lines

        Args:
            file_path: Path to the file to read

        Yields:
            List[str]: Complete lines from each chunk
        """
        file_size = os.path.getsize(file_path)

        if file_size == 0:
            return

        logger.info(f"Reading file {file_path} ({file_size:,} bytes) in chunks of {self.chunk_size_bytes:,} bytes")

        # Use memory mapping for large files if enabled
        if self.config.enable_mmap and file_size > self.chunk_size_bytes * 2:
            yield from self._read_chunks_mmap(file_path, file_size)
        else:
            yield from self._read_chunks_traditional(file_path)

    def _read_chunks_mmap(self, file_path: str, file_size: int) -> Iterator[List[str]]:
        """Read chunks using memory mapping for efficient large file access"""
        with open(file_path, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                position = 0
                leftover = b""

                while position < file_size:
                    # Calculate chunk end position
                    chunk_end = min(position + self.chunk_size_bytes, file_size)

                    # Read chunk data
                    chunk_data = mm[position:chunk_end]

                    # Combine with leftover from previous chunk
                    full_data = leftover + chunk_data

                    # Find line boundaries
                    lines, leftover = self._extract_complete_lines(full_data)

                    if lines:
                        yield [line.decode('utf-8', errors='ignore') for line in lines]

                    position = chunk_end

                # Handle any remaining data
                if leftover:
                    final_lines = leftover.split(b'\n')
                    yield [line.decode('utf-8', errors='ignore') for line in final_lines if line.strip()]

    def _read_chunks_traditional(self, file_path: str) -> Iterator[List[str]]:
        """Read chunks using traditional file I/O"""
        with open(file_path, 'rb') as f:
            leftover = b""

            while True:
                # Read chunk
                chunk_data = f.read(self.chunk_size_bytes)
                if not chunk_data:
                    break

                # Combine with leftover from previous chunk
                full_data = leftover + chunk_data

                # Find line boundaries
                lines, leftover = self._extract_complete_lines(full_data)

                if lines:
                    yield [line.decode('utf-8', errors='ignore') for line in lines]

            # Handle any remaining data
            if leftover:
                final_lines = leftover.split(b'\n')
                yield [line.decode('utf-8', errors='ignore') for line in final_lines if line.strip()]

    def _extract_complete_lines(self, data: bytes) -> Tuple[List[bytes], bytes]:
        """
        Extract complete lines from data, handling partial lines at boundaries

        Args:
            data: Raw bytes to process

        Returns:
            Tuple of (complete_lines, leftover_bytes)
        """
        # Find the last complete line boundary
        last_newline = data.rfind(b'\n')

        if last_newline == -1:
            # No complete lines found
            if len(data) > self.max_line_size_bytes:
                # Line too long, force split (with warning)
                logger.warning(f"Line exceeds max size ({len(data)} bytes), force splitting")
                return [data], b""
            else:
                # Keep as leftover for next chunk
                return [], data

        # Split into complete lines and leftover
        complete_data = data[:last_newline]
        leftover = data[last_newline + 1:]

        # Split complete data into lines
        lines = complete_data.split(b'\n')

        return lines, leftover


class StreamingMapTask:
    """
    Streaming-aware map task executor that processes large files efficiently
    """

    def __init__(self, config: StreamingConfig):
        self.config = config
        self.reader = ChunkedFileReader(config)

    def execute_streaming_map(
        self,
        input_file: str,
        output_file: str,
        map_function: Callable,
        num_reduce_tasks: int
    ) -> bool:
        """
        Execute map task using streaming processing

        Args:
            input_file: Input file path
            output_file: Output file path
            map_function: Map function to apply
            num_reduce_tasks: Number of reduce tasks for partitioning

        Returns:
            bool: Success status
        """
        logger.info(f"Executing streaming map task: {input_file} -> {output_file}")

        total_records = 0

        try:
            with open(output_file, 'w') as out_f:
                # Process file in chunks
                for chunk_lines in self.reader.read_chunks(input_file):
                    # Process each line in the chunk
                    for line_num, line in enumerate(chunk_lines):
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            # Apply map function
                            map_results = map_function(line)

                            # Process map results
                            for key, value in map_results:
                                # Calculate partition
                                partition_id = hash(key) % num_reduce_tasks

                                # Write intermediate record
                                record = {
                                    'key': key,
                                    'value': value,
                                    'partition': partition_id
                                }
                                out_f.write(json.dumps(record) + '\n')
                                total_records += 1

                        except Exception as e:
                            logger.error(f"Error processing line in chunk: {str(e)}")
                            continue

            logger.info(f"Streaming map task completed: {total_records} records written")
            return True

        except Exception as e:
            logger.error(f"Streaming map task failed: {str(e)}")
            return False


class LargeDatasetGenerator:
    """
    Generates large datasets for testing streaming capabilities
    """

    def __init__(self, config: StreamingConfig):
        self.config = config

    def generate_large_text_file(
        self,
        output_path: str,
        size_gb: float,
        content_type: str = "word_count"
    ) -> str:
        """
        Generate a large text file for testing

        Args:
            output_path: Where to write the file
            size_gb: Target file size in GB
            content_type: Type of content ("word_count", "log_analysis", "user_activity")

        Returns:
            str: Path to generated file
        """
        target_size_bytes = int(size_gb * 1024 * 1024 * 1024)

        logger.info(f"Generating {size_gb:.2f}GB test file: {output_path}")

        if content_type == "word_count":
            return self._generate_word_count_data(output_path, target_size_bytes)
        elif content_type == "log_analysis":
            return self._generate_log_data(output_path, target_size_bytes)
        elif content_type == "user_activity":
            return self._generate_user_activity_data(output_path, target_size_bytes)
        else:
            raise ValueError(f"Unknown content type: {content_type}")

    def _generate_word_count_data(self, output_path: str, target_size: int) -> str:
        """Generate large text file with random words"""
        words = [
            'the', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog',
            'hello', 'world', 'mapreduce', 'distributed', 'computing', 'big', 'data',
            'processing', 'scalable', 'systems', 'streaming', 'algorithms'
        ]

        current_size = 0
        lines_written = 0

        with open(output_path, 'w') as f:
            while current_size < target_size:
                # Generate random sentence
                sentence_length = random.randint(5, 20)
                sentence_words = random.choices(words, k=sentence_length)
                line = ' '.join(sentence_words) + '\n'

                f.write(line)
                current_size += len(line.encode('utf-8'))
                lines_written += 1

                # Progress logging
                if lines_written % 100000 == 0:
                    progress = (current_size / target_size) * 100
                    logger.info(f"Generated {lines_written:,} lines ({progress:.1f}%)")

        logger.info(f"Generated {lines_written:,} lines, {current_size:,} bytes")
        return output_path

    def _generate_log_data(self, output_path: str, target_size: int) -> str:
        """Generate large log file with realistic web server logs"""
        ips = ['192.168.1.100', '10.0.0.50', '203.0.113.1', '198.51.100.1']
        methods = ['GET', 'POST', 'PUT', 'DELETE']
        paths = ['/index.html', '/api/users', '/api/products', '/images/logo.png']
        status_codes = [200, 404, 500, 403, 301]

        current_size = 0
        lines_written = 0

        with open(output_path, 'w') as f:
            while current_size < target_size:
                # Generate log entry
                ip = random.choice(ips)
                method = random.choice(methods)
                path = random.choice(paths)
                status = random.choice(status_codes)
                size = random.randint(100, 50000)

                # Simple log format
                line = f'{ip} - - [01/Jan/2024:12:00:00 +0000] "{method} {path} HTTP/1.1" {status} {size}\n'

                f.write(line)
                current_size += len(line.encode('utf-8'))
                lines_written += 1

                if lines_written % 50000 == 0:
                    progress = (current_size / target_size) * 100
                    logger.info(f"Generated {lines_written:,} log entries ({progress:.1f}%)")

        logger.info(f"Generated {lines_written:,} log entries, {current_size:,} bytes")
        return output_path

    def _generate_user_activity_data(self, output_path: str, target_size: int) -> str:
        """Generate large user activity JSON file"""
        import json

        users = [f"user_{i:04d}" for i in range(1000)]
        activities = ['view', 'click', 'like', 'share', 'comment', 'purchase']

        current_size = 0
        lines_written = 0

        with open(output_path, 'w') as f:
            while current_size < target_size:
                # Generate activity event
                event = {
                    'user_id': random.choice(users),
                    'activity_type': random.choice(activities),
                    'timestamp': random.randint(1640995200, 1672531200),  # 2022-2023
                    'value': random.randint(1, 500)
                }

                line = json.dumps(event) + '\n'
                f.write(line)
                current_size += len(line.encode('utf-8'))
                lines_written += 1

                if lines_written % 50000 == 0:
                    progress = (current_size / target_size) * 100
                    logger.info(f"Generated {lines_written:,} events ({progress:.1f}%)")

        logger.info(f"Generated {lines_written:,} events, {current_size:,} bytes")
        return output_path


class StreamingIntegration:
    """
    Integration layer for adding streaming capabilities to existing MapReduce scheduler
    """

    @staticmethod
    def create_streaming_config(
        chunk_size_mb: int = 64,
        enable_mmap: bool = True,
        temp_dir: Optional[str] = None
    ) -> StreamingConfig:
        """Create a streaming configuration"""
        return StreamingConfig(
            chunk_size_mb=chunk_size_mb,
            enable_mmap=enable_mmap,
            temp_dir=temp_dir or tempfile.gettempdir()
        )

    @staticmethod
    def should_use_streaming(file_path: str, threshold_mb: int = 100) -> bool:
        """
        Determine if a file should be processed with streaming

        Args:
            file_path: Path to the file
            threshold_mb: Size threshold in MB for enabling streaming

        Returns:
            bool: True if file should use streaming
        """
        try:
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            return file_size_mb > threshold_mb
        except OSError:
            return False

    @staticmethod
    def split_large_file_for_map_tasks(
        input_file: str,
        num_map_tasks: int,
        output_dir: str,
        config: StreamingConfig
    ) -> List[str]:
        """
        Split a large file into smaller chunks for parallel map processing

        Args:
            input_file: Large input file to split
            num_map_tasks: Number of map tasks to create
            output_dir: Directory for split files
            config: Streaming configuration

        Returns:
            List[str]: Paths to split files
        """
        file_size = os.path.getsize(input_file)
        chunk_size = file_size // num_map_tasks

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        reader = ChunkedFileReader(config)
        split_files = []

        logger.info(f"Splitting {input_file} into {num_map_tasks} chunks")

        current_chunk = 0
        current_size = 0

        chunk_file_path = output_path / f"chunk_{current_chunk:04d}.txt"
        split_files.append(str(chunk_file_path))
        chunk_file = open(chunk_file_path, 'w')

        try:
            for lines in reader.read_chunks(input_file):
                for line in lines:
                    chunk_file.write(line + '\n')
                    current_size += len(line.encode('utf-8'))

                    # Check if we should start a new chunk
                    if current_size >= chunk_size and current_chunk < num_map_tasks - 1:
                        chunk_file.close()
                        current_chunk += 1
                        current_size = 0

                        chunk_file_path = output_path / f"chunk_{current_chunk:04d}.txt"
                        split_files.append(str(chunk_file_path))
                        chunk_file = open(chunk_file_path, 'w')

        finally:
            chunk_file.close()

        logger.info(f"Created {len(split_files)} split files")
        return split_files


# Example usage and testing functions
def test_streaming_capabilities():
    """Test the streaming file processing capabilities"""
    print("🌊 TESTING STREAMING FILE PROCESSING")
    print("=" * 50)

    # Create streaming configuration
    config = StreamingConfig(
        chunk_size_mb=32,  # Smaller chunks for testing
        enable_mmap=True
    )

    # Generate test data
    generator = LargeDatasetGenerator(config)
    test_file = "/tmp/large_test_file.txt"

    print("Creating 100MB test file...")
    generator.generate_large_text_file(test_file, 0.1, "word_count")  # 100MB

    # Test chunked reading
    reader = ChunkedFileReader(config)

    total_lines = 0
    chunk_count = 0

    print("Reading file in chunks...")
    for chunk_lines in reader.read_chunks(test_file):
        chunk_count += 1
        lines_in_chunk = len(chunk_lines)
        total_lines += lines_in_chunk
        print(f"  Chunk {chunk_count}: {lines_in_chunk:,} lines")

        if chunk_count >= 5:  # Limit output for demo
            print("  ...")
            break

    print(f"\nProcessed {total_lines:,} lines in {chunk_count} chunks")
    print("Streaming test completed successfully! ✅")


if __name__ == "__main__":
    test_streaming_capabilities()