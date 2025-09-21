#!/usr/bin/env python3
"""
Comprehensive test runner for the ML framework repository.

This script runs all tests across different modules and provides
detailed reporting. Can be used locally or in CI environments.
"""

import subprocess
import sys
import time


class TestRunner:
    """Orchestrates running all tests with proper reporting."""

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.results = {}
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0

    def log(self, message, level="INFO"):
        """Log message with timestamp."""
        if self.verbose:
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] {level}: {message}")

    def run_command(self, command, cwd=None, description=""):
        """Run a command and capture output."""
        self.log(f"Running: {description or command}")

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode == 0:
                self.log(f"✅ {description or command} - PASSED")
                return True, result.stdout
            else:
                self.log(f"❌ {description or command} - FAILED")
                if self.verbose:
                    self.log(f"STDOUT: {result.stdout}")
                    self.log(f"STDERR: {result.stderr}")
                return False, result.stderr

        except subprocess.TimeoutExpired:
            self.log(f"⏰ {description or command} - TIMEOUT")
            return False, "Command timed out"
        except Exception as e:
            self.log(f"💥 {description or command} - ERROR: {e}")
            return False, str(e)

    def test_autograd_module(self):
        """Test the autograd module comprehensively."""
        self.log("=" * 60)
        self.log("TESTING AUTOGRAD MODULE")
        self.log("=" * 60)

        # Test using custom test runner
        success1, output1 = self.run_command(
            "python run_tests.py", cwd="autograd/tests", description="Autograd custom test suite"
        )

        # Test using pytest
        success2, output2 = self.run_command(
            "python -m pytest tests/ -v", cwd="autograd", description="Autograd pytest suite"
        )

        # Test main script execution
        success3, output3 = self.run_command(
            "python main.py", cwd="autograd", description="Autograd main script"
        )

        self.results["autograd"] = {
            "custom_tests": success1,
            "pytest": success2,
            "main_script": success3,
            "overall": success1 and success2 and success3,
        }

        return self.results["autograd"]["overall"]

    def test_import_functionality(self):
        """Test that all modules can be imported successfully."""
        self.log("=" * 60)
        self.log("TESTING MODULE IMPORTS")
        self.log("=" * 60)

        import_tests = [
            (
                "Autograd imports",
                "import sys; sys.path.append('autograd'); from simple import *; from activations import *; from linear import *",
            ),
            (
                "Lib imports",
                "import sys; sys.path.append('lib'); from activations import *; from loss_functions import *; from utils import *",
            ),
            (
                "Regression imports",
                "import sys; sys.path.append('regression'); from dataset import *; from e_linear_reg import *; from e_non_linear_reg import *",
            ),
            (
                "Classification imports",
                "import sys; sys.path.append('classification'); from dataset import *; from cifar_cnn import *",
            ),
        ]

        all_passed = True
        for description, import_code in import_tests:
            success, output = self.run_command(
                f'python -c "{import_code}"', description=description
            )
            all_passed = all_passed and success

        self.results["imports"] = all_passed
        return all_passed

    def test_regression_module(self):
        """Test regression module functionality."""
        self.log("=" * 60)
        self.log("TESTING REGRESSION MODULE")
        self.log("=" * 60)

        # Use a simpler approach for CircleCI compatibility
        regression_test_file = "temp_regression_test.py"

        with open(regression_test_file, "w") as f:
            f.write(
                """import torch
import numpy as np
import sys
import os
sys.path.append("regression")
sys.path.append(".")
from regression.dataset import generate_polynomial_data
from regression.e_linear_reg import LinearRegressionModel
from regression.e_non_linear_reg import MLP
from regression.configs import RegressionModelConfig

regression_model_config = RegressionModelConfig(
    name="linear",
    custom_act="relu", 
    num_latent_layers=3,
    latent_dims=[16,32,16], 
    allow_residual=False
)
# Test data generation
X, y = generate_polynomial_data(50, degree=2, noise_level=0.1)
print("Generated data: X.shape=" + str(X.shape) + ", y.shape=" + str(y.shape))

# Test linear model
model = LinearRegressionModel(regression_model_config)
pred = model(torch.randn(10, 1))
print("Linear model output shape: " + str(pred.shape))

# Test non-linear model  
mlp_model_config = RegressionModelConfig(
    name="nlinear",
    custom_act="relu", 
    num_latent_layers=3,
    latent_dims=[16,32,16], 
    allow_residual=False
)
nl_model = MLP(mlp_model_config)
nl_pred = nl_model(torch.randn(10, 1))
print("Non-linear model output shape: " + str(nl_pred.shape))

print("✅ All regression tests passed")
"""
            )

        success, output = self.run_command(
            f"python {regression_test_file}", description="Regression module functionality"
        )

        # Clean up
        import os

        try:
            os.unlink(regression_test_file)
        except:
            pass

        self.results["regression"] = success
        return success

    def test_classification_module(self):
        """Test classification module functionality."""
        self.log("=" * 60)
        self.log("TESTING CLASSIFICATION MODULE")
        self.log("=" * 60)

        # Use a simpler approach for CircleCI compatibility
        classification_test_file = "temp_classification_test.py"

        with open(classification_test_file, "w") as f:
            f.write(
                """import torch
import sys
sys.path.append("classification")
from cifar_cnn import CIFARCNN

# Test model creation
model = CIFARCNN(3)
print("Model created: " + type(model).__name__)

# Test forward pass with dummy data
dummy_input = torch.randn(2, 3, 32, 32)  # CIFAR-10 format
output = model(dummy_input)
print("Model output shape: " + str(output.shape))

# Test that output has correct number of classes
assert output.shape[1] == 10, "Expected 10 classes, got " + str(output.shape[1])

print("✅ All classification tests passed")
"""
            )

        success, output = self.run_command(
            f"python {classification_test_file}", description="Classification module functionality"
        )

        # Clean up
        import os

        try:
            os.unlink(classification_test_file)
        except:
            pass

        self.results["classification"] = success
        return success

    def test_mapreduce_module(self):
        """Test the MapReduce module comprehensively."""
        self.log("=" * 60)
        self.log("TESTING MAPREDUCE MODULE")
        self.log("=" * 60)

        # Test using the comprehensive MapReduce test runner
        success1, output1 = self.run_command(
            "python run_mapreduce_tests.py", cwd="mapreduce", description="MapReduce comprehensive test suite"
        )

        # Test main MapReduce framework script
        success2, output2 = self.run_command(
            "python word_stats/map_reduce_framework.py both --data-dir word_stats/data --stats-type word_count",
            cwd="mapreduce", description="MapReduce main framework execution"
        )

        # Test partitioning data generator (quick run)
        success3, output3 = self.run_command(
            "python partitioning/data_generator.py", cwd="mapreduce", description="Partitioning data generator"
        )

        # Test partitioning analyzer
        success4, output4 = self.run_command(
            "python partitioning/partition_analyzer.py", cwd="mapreduce", description="Partitioning analyzer"
        )

        self.results["mapreduce"] = {
            "comprehensive_tests": success1,
            "main_framework": success2,
            "data_generator": success3,
            "partition_analyzer": success4,
            "overall": success1 and success2 and success3 and success4,
        }

        return self.results["mapreduce"]["overall"]

    def test_integration(self):
        """Test cross-module integration."""
        self.log("=" * 60)
        self.log("TESTING CROSS-MODULE INTEGRATION")
        self.log("=" * 60)

        integration_test = """
import torch
import sys
sys.path.append(\\"autograd\\")
sys.path.append(\\"lib\\")

from simple import Square, Cube
from activations import ReLU
from utils import set_seed

# Set seed for reproducibility
set_seed(42)

# Test integration
x = torch.randn(3, 2, requires_grad=True)
y = Square.apply(x)
z = ReLU.apply(y)
loss = z.sum()
loss.backward()

assert x.grad is not None, \\"Gradients not computed\\"
print(\\"✅ Cross-module integration test passed\\")
        """

        success, output = self.run_command(
            f'python -c "{integration_test}"', description="Cross-module integration"
        )

        self.results["integration"] = success
        return success

    def run_all_tests(self):
        """Run comprehensive test suite."""
        self.log("🚀 Starting comprehensive test suite...")
        start_time = time.time()

        # Run all test categories
        test_results = [
            self.test_import_functionality(),
            self.test_autograd_module(),
            self.test_regression_module(),
            self.test_classification_module(),
            self.test_mapreduce_module(),
            self.test_integration(),
        ]

        # Calculate results
        total_categories = len(test_results)
        passed_categories = sum(test_results)

        # Print summary
        elapsed_time = time.time() - start_time
        self.log("=" * 60)
        self.log("TEST SUMMARY")
        self.log("=" * 60)

        for category, result in self.results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            self.log(f"{category.upper()}: {status}")

        self.log(f"\nOverall: {passed_categories}/{total_categories} categories passed")
        self.log(f"Execution time: {elapsed_time:.2f} seconds")

        if passed_categories == total_categories:
            self.log("🎉 ALL TESTS PASSED!")
            return True
        else:
            self.log("💥 SOME TESTS FAILED!")
            return False


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run comprehensive test suite")
    parser.add_argument("--quiet", "-q", action="store_true", help="Reduce output verbosity")
    parser.add_argument(
        "--module",
        "-m",
        choices=["autograd", "regression", "classification", "mapreduce", "imports", "integration"],
        help="Run tests for specific module only",
    )
    args = parser.parse_args()

    runner = TestRunner(verbose=not args.quiet)

    if args.module:
        # Run specific module tests
        if args.module == "autograd":
            success = runner.test_autograd_module()
        elif args.module == "regression":
            success = runner.test_regression_module()
        elif args.module == "classification":
            success = runner.test_classification_module()
        elif args.module == "mapreduce":
            success = runner.test_mapreduce_module()
        elif args.module == "imports":
            success = runner.test_import_functionality()
        elif args.module == "integration":
            success = runner.test_integration()
    else:
        # Run all tests
        success = runner.run_all_tests()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
