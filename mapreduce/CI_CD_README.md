# CI/CD Setup for MapReduce Project

## 🔄 GitHub Actions Integration

The MapReduce project CI/CD pipeline has been integrated with the repository's main GitHub Actions workflow system.

### 📍 **Workflow Location**
- **File**: `/.github/workflows/mapreduce-tests.yml`
- **Scope**: Repository root level (not in mapreduce subdirectory)
- **Working Directory**: All workflow steps run in the `mapreduce/` directory

### 🎯 **Trigger Configuration**

The workflow is optimized to run only when MapReduce-related files change:

```yaml
on:
  push:
    paths:
      - 'mapreduce/**'                    # Any file in mapreduce directory
      - '.github/workflows/mapreduce-tests.yml'  # Workflow file itself
  pull_request:
    paths:
      - 'mapreduce/**'
      - '.github/workflows/mapreduce-tests.yml'
```

### 🚀 **Key Features**

#### **Path-Based Triggering**
- Workflow only runs when MapReduce files are modified
- Reduces unnecessary CI runs for unrelated repository changes
- Includes workflow file itself for testing pipeline changes

#### **Working Directory**
```yaml
defaults:
  run:
    working-directory: mapreduce
```
- All commands execute in the `mapreduce/` subdirectory
- Maintains correct file paths and imports
- Isolates MapReduce project from other repository components

#### **Multi-Job Pipeline**
1. **Test Job**: Unit tests, integration tests, error handling across Python 3.8-3.12
2. **Code Quality Job**: Black, isort, flake8, mypy compliance
3. **Performance Job**: Automated benchmarking with 10k line test files

### 📊 **Workflow Status**

#### **Badge for README**
Add this badge to your main repository README to show MapReduce test status:

```markdown
![MapReduce Tests](https://github.com/USERNAME/REPOSITORY/workflows/MapReduce%20Tests/badge.svg)
```

#### **Viewing Results**
1. Go to repository **Actions** tab
2. Look for **"MapReduce Tests"** workflow
3. Click on runs to see detailed logs for each job

### 🔧 **Local Development**

#### **Running Tests Locally**
```bash
cd mapreduce
python run_tests.py                    # Unit tests
python -m black --check .             # Code formatting
python -m flake8 . --max-line-length=88  # Linting
python -m isort --check-only .         # Import sorting
```

#### **Pre-commit Validation**
Before pushing changes, ensure compliance:
```bash
cd mapreduce
python -m black .           # Auto-format code
python -m isort .            # Sort imports
python run_tests.py         # Run all tests
```

### 📁 **Repository Structure Impact**

```
prep/                           # Repository root
├── .github/
│   └── workflows/
│       ├── mapreduce-tests.yml # ← MapReduce CI/CD pipeline
│       ├── advanced-ci.yml     # Other repository workflows
│       ├── ci.yml
│       ├── pr-checks.yml
│       └── status.yml
├── mapreduce/                  # MapReduce project
│   ├── word_count.py
│   ├── run_tests.py
│   ├── tests/
│   └── requirements.txt
└── other_projects/             # Other repository components
```

### ⚙️ **Configuration Benefits**

#### **Isolation**
- MapReduce tests don't interfere with other project CI/CD
- Path-based triggering reduces CI noise
- Independent test environment and dependencies

#### **Integration**
- Centralized workflow management
- Consistent CI/CD patterns across repository
- Shared infrastructure and resources

#### **Efficiency**
- Only runs when relevant files change
- Parallel job execution across test matrix
- Optimized for development workflow

### 🔍 **Monitoring and Maintenance**

#### **Workflow Health**
- Monitor test success rates in Actions tab
- Review performance trends over time
- Update Python version matrix as needed

#### **Path Updates**
If you move or rename the mapreduce directory:
```yaml
# Update paths in workflow file
paths:
  - 'new_mapreduce_location/**'

defaults:
  run:
    working-directory: new_mapreduce_location
```

### 📚 **Related Documentation**

- **Main README**: `/mapreduce/README.md` - Project overview and usage
- **Test Documentation**: `/mapreduce/tests/README.md` - Test structure and guidelines
- **Requirements**: `/mapreduce/requirements.txt` - Dependencies and versions

### 🆘 **Troubleshooting**

#### **Workflow Not Triggering**
- Verify file changes are in `mapreduce/**` paths
- Check that branch names match trigger configuration
- Ensure GitHub Actions are enabled for the repository

#### **Tests Failing in CI But Passing Locally**
- Check Python version differences (CI tests multiple versions)
- Verify working directory is set correctly
- Review file path dependencies

#### **Performance Test Failures**
- GitHub Actions have resource limitations
- Adjust performance thresholds if needed
- Consider using smaller test datasets for CI

---

**✅ The MapReduce project is now fully integrated with the repository's CI/CD infrastructure while maintaining clean separation and efficient resource usage!**