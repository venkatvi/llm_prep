"""
Copyright (c) 2025. All rights reserved.
"""

"""
Test runner for transformer library tests.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
import torch


def run_transformer_tests():
    """Run all transformer tests with proper configuration."""
    
    # Set torch to deterministic mode for reproducible tests
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    # Configure test parameters
    test_args = [
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--strict-markers",  # Strict marker checking
        os.path.dirname(__file__),  # Test directory
    ]
    
    print("🧪 Running Transformer Library Tests...")
    print("=" * 50)
    
    # Run tests
    exit_code = pytest.main(test_args)
    
    if exit_code == 0:
        print("\n✅ All transformer tests passed!")
    else:
        print(f"\n❌ Some tests failed (exit code: {exit_code})")
    
    return exit_code


def run_individual_test_files():
    """Run each test file individually for detailed output."""
    
    test_files = [
        "test_attention.py",
        "test_ffn.py", 
        "test_input_encodings.py",
        "test_transformer_model.py",
        "test_integration.py"
    ]
    
    results = {}
    test_dir = os.path.dirname(__file__)
    
    for test_file in test_files:
        print(f"\n🔍 Running {test_file}...")
        print("-" * 40)
        
        test_path = os.path.join(test_dir, test_file)
        exit_code = pytest.main(["-v", test_path])
        
        results[test_file] = exit_code
        
        if exit_code == 0:
            print(f"✅ {test_file}: PASSED")
        else:
            print(f"❌ {test_file}: FAILED")
    
    print("\n📊 Test Summary:")
    print("=" * 50)
    
    passed = sum(1 for code in results.values() if code == 0)
    total = len(results)
    
    for test_file, exit_code in results.items():
        status = "PASSED" if exit_code == 0 else "FAILED"
        print(f"{test_file:30s} {status}")
    
    print(f"\nOverall: {passed}/{total} test files passed")
    
    return all(code == 0 for code in results.values())


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run transformer tests")
    parser.add_argument(
        "--individual", 
        action="store_true", 
        help="Run each test file individually"
    )
    
    args = parser.parse_args()
    
    if args.individual:
        success = run_individual_test_files()
    else:
        success = run_transformer_tests() == 0
    
    sys.exit(0 if success else 1)