#!/usr/bin/env python3
"""
Comprehensive Test Runner for MapReduce Framework

This script runs all test suites for the complete MapReduce implementation
including core functionality, extended classes, factory patterns, performance
tests, and integration tests.

Usage:
    python run_mapreduce_tests.py                    # Run all tests
    python run_mapreduce_tests.py --verbose          # Run with detailed output
    python run_mapreduce_tests.py --test-suite NAME  # Run specific test suite
"""

import sys
import time
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple


class TestRunner:
    """Manages and executes all test suites for the MapReduce framework."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.word_stats_dir = Path(__file__).parent / "word_stats"
        self.test_suites = {
            "core": {
                "file": "tests/test_word_count.py",
                "description": "Core MapReduce functionality tests"
            },
            "classes": {
                "file": "tests/test_mapreduce_classes.py",
                "description": "MapReduce class implementation tests"
            },
            "factory": {
                "file": "tests/test_factory_pattern.py",
                "description": "Factory pattern and registry tests"
            },
            "extended": {
                "file": "tests/test_extended_mapreduce_classes.py",
                "description": "Extended MapReduce classes (TopK, FreqCount)"
            },
            "performance": {
                "file": "tests/test_performance_edge_cases.py",
                "description": "Performance and edge case tests"
            },
            "integration": {
                "file": "tests/test_integration_framework.py",
                "description": "Full framework integration tests"
            }
        }

    def run_test_suite(self, suite_name: str, test_info: Dict) -> Tuple[bool, float, str]:
        """Run a single test suite and return results."""
        test_file = self.word_stats_dir / test_info["file"]

        if not test_file.exists():
            return False, 0.0, f"Test file not found: {test_file}"

        if self.verbose:
            print(f"\n🔄 Running {suite_name}: {test_info['description']}")

        start_time = time.time()

        try:
            # Run the test file as a subprocess to capture output
            result = subprocess.run(
                [sys.executable, str(test_file)],
                cwd=str(self.word_stats_dir),
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout per test suite
            )

            execution_time = time.time() - start_time

            if result.returncode == 0:
                # Parse success from output
                output = result.stdout
                if "🎉" in output and ("All" in output and "passed" in output):
                    if self.verbose:
                        print(f"✅ {suite_name} passed ({execution_time:.2f}s)")
                    return True, execution_time, output
                else:
                    if self.verbose:
                        print(f"⚠️  {suite_name} completed but may have issues")
                    return False, execution_time, output
            else:
                error_msg = result.stderr or result.stdout
                if self.verbose:
                    print(f"❌ {suite_name} failed ({execution_time:.2f}s)")
                    print(f"Error: {error_msg}")
                return False, execution_time, error_msg

        except subprocess.TimeoutExpired:
            return False, 120.0, "Test timed out after 2 minutes"
        except Exception as e:
            return False, time.time() - start_time, f"Exception: {str(e)}"

    def run_all_tests(self, specific_suite: str = None) -> bool:
        """Run all test suites or a specific suite."""
        print("🚀 COMPREHENSIVE MAPREDUCE TEST RUNNER")
        print("=" * 60)

        if not self.word_stats_dir.exists():
            print(f"❌ Error: word_stats directory not found at {self.word_stats_dir}")
            return False

        # Determine which suites to run
        if specific_suite:
            if specific_suite not in self.test_suites:
                print(f"❌ Unknown test suite: {specific_suite}")
                print(f"Available suites: {', '.join(self.test_suites.keys())}")
                return False
            suites_to_run = {specific_suite: self.test_suites[specific_suite]}
        else:
            suites_to_run = self.test_suites

        # Run tests
        results = {}
        total_time = 0
        failed_suites = []

        for suite_name, test_info in suites_to_run.items():
            success, exec_time, output = self.run_test_suite(suite_name, test_info)
            results[suite_name] = {
                'success': success,
                'time': exec_time,
                'output': output,
                'description': test_info['description']
            }
            total_time += exec_time

            if not success:
                failed_suites.append(suite_name)

        # Print summary
        self._print_summary(results, total_time, failed_suites)

        return len(failed_suites) == 0

    def _print_summary(self, results: Dict, total_time: float, failed_suites: List[str]):
        """Print comprehensive test summary."""
        print("\n" + "=" * 60)
        print("TEST EXECUTION SUMMARY")
        print("=" * 60)

        passed = 0
        total = len(results)

        for suite_name, result in results.items():
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            time_str = f"{result['time']:.2f}s"
            print(f"{status:<10} {suite_name:<12} {time_str:<8} - {result['description']}")

            if result['success']:
                passed += 1
            elif self.verbose:
                print(f"         Error details: {result['output'][:200]}...")

        print("-" * 60)
        print(f"Results: {passed}/{total} test suites passed")
        print(f"Total execution time: {total_time:.2f} seconds")

        if failed_suites:
            print(f"\n❌ Failed test suites: {', '.join(failed_suites)}")
            print("💡 Run with --verbose for detailed error information")
            print("💡 Run individual suites with --test-suite <name>")
        else:
            print("\n🎉 ALL TEST SUITES PASSED!")
            print("✨ Your MapReduce framework is working perfectly!")

        print("=" * 60)

    def list_test_suites(self):
        """List all available test suites."""
        print("Available test suites:")
        print("-" * 40)
        for name, info in self.test_suites.items():
            print(f"  {name:<12} - {info['description']}")


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="Comprehensive test runner for MapReduce framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_mapreduce_tests.py                    # Run all tests
  python run_mapreduce_tests.py --verbose          # Run with detailed output
  python run_mapreduce_tests.py --test-suite core  # Run only core tests
  python run_mapreduce_tests.py --list            # List available test suites
        """
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output with detailed test information"
    )

    parser.add_argument(
        "--test-suite", "-t",
        type=str,
        help="Run specific test suite (core, classes, factory, extended, performance, integration)"
    )

    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available test suites"
    )

    args = parser.parse_args()

    runner = TestRunner(verbose=args.verbose)

    if args.list:
        runner.list_test_suites()
        return 0

    # Run tests
    success = runner.run_all_tests(args.test_suite)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())