# PM2.5 Air Quality Prediction - Copilot Instructions

## Project Overview

This is a Python project that predicts PM2.5 air quality levels in India using a two-stage machine learning model architecture. The system processes satellite data from various sources (TROPOMI, MERRA, ERA5, Google Earth Engine) combined with air quality station data to generate daily predictions at 10km resolution.

**Repository Stats:**
- ~17K lines of Python code across 128 files
- Python 3.12 project using Poetry for dependency management
- Docker containerized with Google Cloud deployment
- Uses modern data stack: Polars, xarray, scikit-learn, XGBoost, LightGBM

**Core ML Architecture:**
1. **Stage 1**: Imputation models for satellite data (AOD, NO2, CO) to fill missing values
2. **Stage 2**: PM2.5 prediction using combined satellite features + station data

## Build & Development Commands

**Always run these commands in this exact order:**

### 1. Environment Setup
```bash
# Install Poetry (if not available)
pip install poetry

# Install dependencies - ALWAYS use these specific groups
poetry install --only main,dev
```
**Critical:** Never add GDAL dependencies - this project specifically avoids GDAL to simplify deployment.

### 2. Testing
```bash
# Run unit tests (takes ~5 seconds)
poetry run pytest -m "not integration"

# Run integration tests (requires GCP auth + environment variables)
poetry run pytest -m "integration"

# Integration tests require these environment variables:
# - IT_GEE_ASSET_BUCKET_NAME
# - IT_GEE_ASSET_ROOT
# - GEE_SERVICE_ACCOUNT_KEY
```

### 3. Code Quality
```bash
# Run type checking
poetry run mypy src/pm25ml

# Run linting
poetry run ruff check src/pm25ml

# Run formatting
poetry run ruff format src/pm25ml

# Check integration test marks
python .ci/scripts/check_integration_marks.py
```

### 4. Build
```bash
# Build package
poetry build

# Build Docker image
docker build -t pm25ml .
```

**Build Time Expectations:**
- `poetry install`: 2-3 minutes (first time)
- Unit tests: ~5 seconds
- Type checking: ~10 seconds
- Linting: ~2 seconds
- Package build: ~5 seconds

## Project Architecture & Layout

**Main Source Structure:**
```
src/pm25ml/
├── collectors/          # Data collection from various sources
│   ├── gee/            # Google Earth Engine data collection
│   ├── ned/            # NASA Earth Data collection
│   └── misc/           # Miscellaneous data sources
├── combiners/          # Data combination and merging logic
├── feature_generation/ # Feature engineering
├── imputation/         # ML models for imputing missing satellite data
├── training/           # PM2.5 model training pipeline
├── sample/             # Data sampling for model training
└── run/                # Entry points for different workflow stages
```

**Configuration Files:**
- `pyproject.toml`: Poetry dependencies, pytest config, ruff config, mypy config
- `.pre-commit-config.yaml`: Pre-commit hooks for ruff + integration test validation
- `Dockerfile`: Multi-stage build (deps → runtime)
- `docker-compose.yml`: Local development setup
- `infra/`: Google Cloud Workflows and Cloud Run job definitions

**Key Dependencies:**
- **Data Processing**: polars (primary), xarray (raw data), pyarrow (parquet)
- **ML**: scikit-learn, xgboost, lightgbm
- **Cloud**: earthengine-api, gcsfs, earthaccess
- **Geospatial**: shapely, pyproj, pyshp (NO GDAL!)

## CI/CD & Validation

**GitHub Actions Workflow (.github/workflows/ci.yml):**
1. **deps**: Cache Poetry and dependencies
2. **install-package**: Test production installation
3. **test**: Run unit tests
4. **integration-test**: Run integration tests with GCP auth
5. **mypy**: Type checking
6. **no-gdal-check**: Ensure GDAL is not in dependencies
7. **lint**: Ruff linting
8. **build-image**: Docker build and push to Google Artifact Registry
9. **release**: Deploy to Google Cloud (on main branch)

**Pre-commit Hooks:**
- Ruff formatting and linting
- Integration test mark validation (ensures all `*__it.py` tests have `@pytest.mark.integration`)

**Deployment Environment:**
- Google Cloud Run Jobs for data processing
- Google Cloud Workflows for orchestration
- Google Cloud Batch for compute-intensive tasks
- Artifact Registry for container images
- Cloud Storage for data

## Testing Conventions

**File Naming:**
- Unit tests: `<module_name>__test.py`
- Integration tests: `<module_name>__it.py`

**Function Naming:**
```python
def test__<thing_under_test>__<situation_under_test>__<expected_outcome>():
```

**Integration Test Requirements:**
- All integration tests MUST be marked with `@pytest.mark.integration`
- Integration tests require GCP authentication and specific environment variables
- Use `pytestmark = pytest.mark.integration` at module level if all tests are integration tests

**Test Execution:**
- Unit tests run in ~5 seconds, no external dependencies
- Integration tests require cloud services and take longer

## Code Standards

**Type System:**
- Strict mypy configuration with type hints required
- Exclusions for external libraries without types (fsspec, gcsfs, earthaccess, etc.)

**Linting:**
- Uses ruff "ALL" rules configuration 
- 100 character line length
- Test files excluded from linting
- Specific ignores: D203, D212 (docstring formatting conflicts)

**Code Style:**
- Use Polars for all data manipulation (not pandas)
- Use xarray for raw geospatial data processing  
- Type all function signatures
- Follow existing patterns in the codebase

## Entry Points & Workflow Stages

The system runs via these main entry points in `src/pm25ml/run/`:
- `fetch_and_combine.py`: Data collection and initial combination
- `generate_features.py`: Feature engineering pipeline
- `sample_for_imputation.py`: Create training samples for imputation models
- `train_*_imputer.py`: Train individual imputation models (AOD, CO, NO2)

**Data Flow:**
1. Collect satellite/station data → Store in GCS buckets with Hive partitioning
2. Combine data sources → Generate features → Sample for training
3. Train imputation models → Apply imputation → Train PM2.5 model → Generate predictions

## Trust These Instructions

These instructions are comprehensive and tested. Only search for additional information if you encounter errors or if specific implementation details are missing. The build and test commands listed here are verified to work correctly.
