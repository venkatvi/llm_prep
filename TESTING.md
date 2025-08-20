# Testing and CI/CD Setup

Testing infrastructure for the ML Framework repository.

## Test Structure
```
├── .github/workflows/     # CI/CD pipelines
├── autograd/tests/        # Autograd tests (72 tests)
├── run_all_tests.py       # Test orchestrator
├── pytest.ini            # Pytest config
└── Makefile               # Development commands
```

## Quick Start

```bash
# Run tests
make test                          # All tests
python run_all_tests.py           # Comprehensive suite
cd autograd/tests && python run_tests.py  # Autograd only

# Development
make format                        # Code formatting
make lint                          # Linting
make setup-pre-commit             # Pre-commit hooks
```

## 🔄 CI/CD Pipelines

### Main CI Pipeline (`ci.yml`)

**Triggers:**
- Push to `master`, `main`, `develop` branches
- Pull requests to `master`, `main`, `develop`
- Manual trigger via workflow_dispatch

**Features:**
- **Multi-Python Testing**: Tests on Python 3.9, 3.10, 3.11, 3.12
- **Dependency Caching**: Speeds up builds with pip cache
- **Comprehensive Testing**: All modules tested automatically
- **Code Quality**: Linting with flake8, formatting with black
- **Integration Tests**: Cross-module functionality verification

**Test Matrix:**
```yaml
strategy:
  matrix:
    python-version: ['3.9', '3.10', '3.11', '3.12']
```

### Advanced CI Pipeline (`advanced-ci.yml`)

**Additional Features:**
- **Path-based Testing**: Only tests changed modules
- **Coverage Reporting**: Code coverage with Codecov integration
- **Performance Benchmarks**: Automated performance testing
- **Security Scanning**: Vulnerability checks with safety and bandit
- **Documentation Building**: Automatic docs generation

**Smart Testing:**
```yaml
# Only test autograd if autograd files changed
if: ${{ needs.changes.outputs.autograd == 'true' }}
```

### Pull Request Checks (`pr-checks.yml`)

**Quick Validation:**
- Fast import tests (< 1 minute)
- Essential autograd tests
- Code formatting checks
- Automatic PR comments with results

## 🔧 Pre-commit Hooks

Automatically run before each commit:

```yaml
# Code formatting
- black (code formatting)
- isort (import sorting)
- flake8 (linting)

# Quality checks
- trailing-whitespace
- end-of-file-fixer
- check-yaml, check-json
- debug-statements

# Custom tests
- autograd-tests (run autograd test suite)
- import-tests (verify all imports work)
```

Setup:
```bash
make setup-pre-commit
# or
pre-commit install
```

## 📊 Test Categories

### 1. Unit Tests
- **Location**: `autograd/tests/`
- **Coverage**: All custom autograd functions
- **Examples**: 
  - Mathematical functions (Power, Square, Cube, Exp)
  - Neural network layers (Linear)
  - Activation functions (Tanh, Sigmoid, ReLU, LearnedSiLU)

### 2. Integration Tests
- **Cross-module functionality**
- **End-to-end pipeline testing**
- **Gradient flow verification**

### 3. Import Tests
- **Module importability**
- **Dependency verification**
- **Quick smoke tests**

### 4. Performance Tests
- **Benchmark autograd functions**
- **Memory usage monitoring**
- **Execution time tracking**

## 📈 Test Metrics

### Current Test Coverage
- **72 total tests** in autograd module
- **100% pass rate** maintained
- **4 test categories**: simple, linear, activations, main
- **Multiple test frameworks**: Custom runner + pytest

### Performance Benchmarks
- Power function: ~0.001s per operation
- Linear layer: ~0.002s per forward+backward
- All activations: < 0.001s per operation

## 🛠️ Development Workflow

### Before Committing
1. **Run tests**: `make test`
2. **Format code**: `make format`
3. **Check linting**: `make lint`
4. **Commit** (pre-commit hooks run automatically)

### Pull Request Process
1. **Create PR** → Triggers `pr-checks.yml`
2. **Quick validation** (< 2 minutes)
3. **Review feedback** via automated comments
4. **Merge** → Triggers full CI pipeline

### Debugging Failed Tests

```bash
# Check specific module
python run_all_tests.py --module autograd

# Run with verbose output
python run_all_tests.py --verbose

# Check imports only
python run_all_tests.py --module imports

# Manual test execution
cd autograd/tests && python run_tests.py
```

## 🎯 Quality Gates

### Required for Merge
- ✅ All import tests pass
- ✅ Autograd test suite passes (72/72)
- ✅ No linting errors
- ✅ Code properly formatted

### Additional Checks (Non-blocking)
- 📊 Performance benchmarks
- 🔒 Security scans
- 📚 Documentation builds
- 🧹 Code quality metrics

## 🔍 Monitoring and Alerts

### GitHub Actions Status
- **Status badges** show current build status
- **Email notifications** for failed builds (maintainers)
- **PR comments** provide immediate feedback

### Coverage Tracking
- **Codecov integration** for coverage reports
- **Coverage trends** tracked over time
- **Coverage requirements** enforced

## 🚨 Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Verify dependencies
pip install -r requirements.txt
```

**Test Failures:**
```bash
# Run specific test
cd autograd && python -m pytest tests/test_simple.py -v

# Check test environment
python run_all_tests.py --module imports
```

**CI/CD Issues:**
- Check `.github/workflows/` syntax
- Verify secrets/environment variables
- Review GitHub Actions logs

### Getting Help

1. **Check test logs** in GitHub Actions
2. **Run tests locally** with verbose output
3. **Review TESTING.md** for guidance
4. **Check issue tracker** for known problems

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Pytest Documentation](https://docs.pytest.org/)
- [Pre-commit Documentation](https://pre-commit.com/)
- [Black Code Formatter](https://black.readthedocs.io/)

---

*This testing infrastructure ensures code quality, prevents regressions, and enables confident continuous deployment.*