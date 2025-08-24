"""
Copyright (c) 2025. All rights reserved.
"""

"""
Unit tests for regression dataset functionality.
"""

import os
import sys
import tempfile
import unittest

import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from regression.dataset import (
    RegressionDataset,
    generate_data_as_csv,
    generate_polynomial_data,
    get_dataloader,
    prepare_data,
)


class TestRegressionDataset(unittest.TestCase):
    """Test suite for RegressionDataset class."""

    def setUp(self):
        """Set up test fixtures with temporary CSV files."""
        # Create a temporary CSV file for testing
        self.temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        )
        self.temp_file_path = self.temp_file.name

        # Write test data
        test_data = {
            "inputs": [1.0, 2.0, 3.0, 4.0, 5.0],
            "targets": [2.0, 4.0, 6.0, 8.0, 10.0],
        }  # y = 2*x
        df = pd.DataFrame(test_data)
        df.to_csv(self.temp_file_path, index=False)
        self.temp_file.close()

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_file_path):
            os.unlink(self.temp_file_path)

    def test_init_valid_file(self):
        """Test initialization with valid CSV file."""
        dataset = RegressionDataset(self.temp_file_path)

        self.assertEqual(len(dataset), 5)
        self.assertEqual(dataset.inputs.shape, (5, 1))
        self.assertEqual(dataset.targets.shape, (5, 1))

        # Check data values
        expected_inputs = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
        expected_targets = torch.tensor([[2.0], [4.0], [6.0], [8.0], [10.0]])

        torch.testing.assert_close(dataset.inputs, expected_inputs)
        torch.testing.assert_close(dataset.targets, expected_targets)

    def test_getitem(self):
        """Test __getitem__ method."""
        dataset = RegressionDataset(self.temp_file_path)

        for i in range(len(dataset)):
            input_val, target_val = dataset[i]
            self.assertEqual(input_val.shape, (1,))
            self.assertEqual(target_val.shape, (1,))

        # Test specific values
        input_val, target_val = dataset[0]
        self.assertAlmostEqual(input_val.item(), 1.0)
        self.assertAlmostEqual(target_val.item(), 2.0)

    def test_invalid_file(self):
        """Test with non-existent file."""
        with self.assertRaises(FileNotFoundError):
            RegressionDataset("nonexistent_file.csv")

    def test_invalid_columns(self):
        """Test with CSV missing required columns."""
        # Create CSV with wrong columns
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        wrong_data = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
        wrong_data.to_csv(temp_file.name, index=False)
        temp_file.close()

        try:
            with self.assertRaises(KeyError):
                RegressionDataset(temp_file.name)
        finally:
            os.unlink(temp_file.name)


class TestDatasetFunctions(unittest.TestCase):
    """Test suite for dataset utility functions."""

    def test_generate_polynomial_data_linear(self):
        """Test linear polynomial data generation."""
        inputs, targets = generate_polynomial_data(
            num_samples=100, degree=1, noise_level=0.0, random_seed=42
        )

        self.assertEqual(inputs.shape, (100, 1))
        self.assertEqual(targets.shape, (100, 1))

        # Check that all values are finite
        self.assertTrue(torch.isfinite(inputs).all())
        self.assertTrue(torch.isfinite(targets).all())

    def test_generate_polynomial_data_quadratic(self):
        """Test quadratic polynomial data generation."""
        inputs, targets = generate_polynomial_data(
            num_samples=50, degree=2, noise_level=0.1, random_seed=123
        )

        self.assertEqual(inputs.shape, (50, 1))
        self.assertEqual(targets.shape, (50, 1))

        # Check that values are in expected range
        self.assertTrue(torch.all(inputs >= 0.0))
        self.assertTrue(torch.all(inputs <= 10.0))

    def test_generate_polynomial_data_with_custom_range(self):
        """Test polynomial generation with custom input range."""
        inputs, targets = generate_polynomial_data(
            num_samples=30, degree=1, x_range=(-5.0, 5.0), random_seed=456
        )

        self.assertTrue(torch.all(inputs >= -5.0))
        self.assertTrue(torch.all(inputs <= 5.0))

    def test_generate_polynomial_data_reproducibility(self):
        """Test that same seed produces same results."""
        inputs1, targets1 = generate_polynomial_data(num_samples=20, random_seed=42)
        inputs2, targets2 = generate_polynomial_data(num_samples=20, random_seed=42)

        torch.testing.assert_close(inputs1, inputs2)
        torch.testing.assert_close(targets1, targets2)

    def test_generate_data_as_csv(self):
        """Test CSV file generation from tensors."""
        inputs = torch.tensor([[1.0], [2.0], [3.0]])
        targets = torch.tensor([[2.0], [4.0], [6.0]])

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            generate_data_as_csv(inputs, targets, temp_path)

            # Verify file exists and has correct content
            self.assertTrue(os.path.exists(temp_path))

            df = pd.read_csv(temp_path)
            self.assertEqual(list(df.columns), ["inputs", "targets"])
            self.assertEqual(len(df), 3)

            # Check values
            expected_inputs = [1.0, 2.0, 3.0]
            expected_targets = [2.0, 4.0, 6.0]

            self.assertEqual(df["inputs"].tolist(), expected_inputs)
            self.assertEqual(df["targets"].tolist(), expected_targets)

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_get_dataloader(self):
        """Test DataLoader creation from CSV file."""
        # Create temporary CSV
        inputs = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        targets = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            generate_data_as_csv(inputs, targets, temp_path)

            # Test different batch sizes
            for batch_size in [1, 2, 4]:
                with self.subTest(batch_size=batch_size):
                    dataloader = get_dataloader(temp_path, batch_size=batch_size)

                    # Check batch size
                    for batch_inputs, batch_targets in dataloader:
                        self.assertLessEqual(len(batch_inputs), batch_size)
                        self.assertEqual(batch_inputs.shape[1], 1)
                        self.assertEqual(batch_targets.shape[1], 1)
                        break  # Just test first batch

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_prepare_data(self):
        """Test prepare_data function that creates temp CSV and DataLoader."""
        inputs = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
        targets = torch.tensor([[2.0], [4.0], [6.0], [8.0], [10.0]])

        dataloader, csv_path = prepare_data(inputs, targets, ".csv", batch_size=2)

        try:
            # Check that CSV file was created
            self.assertTrue(os.path.exists(csv_path))

            # Check DataLoader functionality
            batch_count = 0
            total_samples = 0

            for batch_inputs, batch_targets in dataloader:
                batch_count += 1
                total_samples += len(batch_inputs)

                # Check shapes
                self.assertEqual(batch_inputs.shape[1], 1)
                self.assertEqual(batch_targets.shape[1], 1)
                self.assertLessEqual(len(batch_inputs), 2)  # batch_size = 2

            self.assertEqual(total_samples, 5)  # All samples accounted for

        finally:
            # Clean up
            if os.path.exists(csv_path):
                os.unlink(csv_path)

    def test_edge_cases(self):
        """Test edge cases for dataset functions."""
        # Test with single sample
        inputs, targets = generate_polynomial_data(num_samples=1, random_seed=42)
        self.assertEqual(inputs.shape, (1, 1))
        self.assertEqual(targets.shape, (1, 1))

        # Test with zero noise
        inputs, targets = generate_polynomial_data(
            num_samples=10, noise_level=0.0, random_seed=42
        )
        # With same seed and no noise, targets should be deterministic
        inputs2, targets2 = generate_polynomial_data(
            num_samples=10, noise_level=0.0, random_seed=42
        )
        torch.testing.assert_close(targets, targets2)


if __name__ == "__main__":
    unittest.main()
