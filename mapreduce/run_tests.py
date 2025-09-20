#!/usr/bin/env python3
"""
Test runner for MapReduce implementation.

This script runs all unit tests for the MapReduce word count implementation
and provides detailed test results and coverage information.

Usage:
    python run_tests.py
    python -m pytest tests/  # Alternative using pytest if installed
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from tests.test_word_count import run_all_tests


def main():
    """
    Main test runner function.

    Executes all tests and exits with appropriate status code.
    """
    print("=" * 60)
    print("MAPREDUCE TEST RUNNER")
    print("=" * 60)
    print("Running comprehensive test suite for MapReduce implementation...")

    # Run all tests
    success = run_all_tests()

    if success:
        print("\n🎉 All tests completed successfully!")
        print("✅ MapReduce implementation is working correctly.")
        return 0
    else:
        print("\n❌ Some tests failed!")
        print("🔧 Please review the failed tests and fix the issues.")
        return 1


if __name__ == "__main__":
    """
    Execute test runner when script is run directly.
    """
    sys.exit(main())
