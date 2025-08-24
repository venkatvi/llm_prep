"""
Copyright (c) 2025. All rights reserved.
"""

"""
Test runner for all autograd tests.

This script runs all test suites for the custom autograd implementation
and provides a comprehensive test report with coverage of all functions.
"""

import os
import sys
import unittest

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all test modules
from test_activations import (
    TestActivationConsistency,
    TestActivationEdgeCases,
    TestLearnedSiLU,
    TestReLU,
    TestSigmoid,
    TestTanh,
)
from test_linear import TestLinear, TestLinearEdgeCases
from test_main import TestIntegration, TestMainScript
from test_simple import TestCube, TestExp, TestPower, TestSquare, TestTensorOperations


def create_test_suite() -> unittest.TestSuite:
    """Create a comprehensive test suite for all autograd functions"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add simple function tests
    suite.addTests(loader.loadTestsFromTestCase(TestPower))
    suite.addTests(loader.loadTestsFromTestCase(TestSquare))
    suite.addTests(loader.loadTestsFromTestCase(TestCube))
    suite.addTests(loader.loadTestsFromTestCase(TestExp))
    suite.addTests(loader.loadTestsFromTestCase(TestTensorOperations))

    # Add linear layer tests
    suite.addTests(loader.loadTestsFromTestCase(TestLinear))
    suite.addTests(loader.loadTestsFromTestCase(TestLinearEdgeCases))

    # Add activation function tests
    suite.addTests(loader.loadTestsFromTestCase(TestTanh))
    suite.addTests(loader.loadTestsFromTestCase(TestSigmoid))
    suite.addTests(loader.loadTestsFromTestCase(TestReLU))
    suite.addTests(loader.loadTestsFromTestCase(TestLearnedSiLU))
    suite.addTests(loader.loadTestsFromTestCase(TestActivationConsistency))
    suite.addTests(loader.loadTestsFromTestCase(TestActivationEdgeCases))

    # Add integration tests
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestMainScript))

    return suite


def run_specific_test_category(category: str) -> bool:
    """Run tests for a specific category"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    if category == "simple":
        suite.addTests(loader.loadTestsFromTestCase(TestPower))
        suite.addTests(loader.loadTestsFromTestCase(TestSquare))
        suite.addTests(loader.loadTestsFromTestCase(TestCube))
        suite.addTests(loader.loadTestsFromTestCase(TestExp))
        suite.addTests(loader.loadTestsFromTestCase(TestTensorOperations))
    elif category == "linear":
        suite.addTests(loader.loadTestsFromTestCase(TestLinear))
        suite.addTests(loader.loadTestsFromTestCase(TestLinearEdgeCases))
    elif category == "activations":
        suite.addTests(loader.loadTestsFromTestCase(TestTanh))
        suite.addTests(loader.loadTestsFromTestCase(TestSigmoid))
        suite.addTests(loader.loadTestsFromTestCase(TestReLU))
        suite.addTests(loader.loadTestsFromTestCase(TestLearnedSiLU))
        suite.addTests(loader.loadTestsFromTestCase(TestActivationConsistency))
        suite.addTests(loader.loadTestsFromTestCase(TestActivationEdgeCases))
    elif category == "integration":
        suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
        suite.addTests(loader.loadTestsFromTestCase(TestMainScript))
    else:
        print(f"Unknown category: {category}")
        print("Available categories: simple, linear, activations, integration")
        return False

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


def main() -> int:
    """Main test runner function"""
    import argparse

    parser = argparse.ArgumentParser(description="Run autograd tests")
    parser.add_argument(
        "--category",
        type=str,
        choices=["simple", "linear", "activations", "integration"],
        help="Run tests for specific category only",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Run tests with verbose output")

    args = parser.parse_args()

    print("=" * 70)
    print("PYTORCH CUSTOM AUTOGRAD TEST SUITE")
    print("=" * 70)

    if args.category:
        print(f"Running {args.category} tests...")
        success = run_specific_test_category(args.category)
    else:
        print("Running all tests...")
        suite = create_test_suite()
        runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
        result = runner.run(suite)
        success = result.wasSuccessful()

    print("\n" + "=" * 70)
    if success:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED!")
    print("=" * 70)

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
