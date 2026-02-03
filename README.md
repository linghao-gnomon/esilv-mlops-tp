# Titanic Survival Prediction

A machine learning project for predicting passenger survival on the Titanic using Random Forest classification.

## Project Structure

```
.
├── data/
│   ├── raw/           # Raw data files
│   └── processed/     # Processed data files
├── notebooks/         # Jupyter notebooks for exploration
├── src/               # Source code
│   ├── data/          # Data utilities
│   │   ├── load.py    # Data loading
│   │   └── split.py   # Data splitting
│   ├── model.py       # Model creation and training
│   ├── evaluation.py  # Model evaluation
│   ├── constants.py   # Project constants
│   └── main.py        # Main entry point
├── tests/             # Unit tests
│   ├── test_data.py   # Data loading tests
│   ├── test_model.py  # Model tests
│   └── test_training.py # Training tests
├── .github/
│   └── workflows/     # GitHub Actions CI/CD
├── Makefile           # Common tasks
├── Dockerfile         # Docker image definition
├── pyproject.toml     # Project dependencies
└── requirements.txt   # Python dependencies (alternative)
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd esilv-mlops-tp
```

2. Install dependencies using `uv`:
```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync
```

3. Configure environment (optional):
```bash
cp example.env .env
# Edit .env with your configuration
```

## Usage

### Run the pipeline

```bash
# Using Makefile (default: 20 trees)
make run

# Or directly with uv
uv run python src/main.py

# Or with standard python (after uv sync)
python src/main.py
```

### Adjust model parameters

```bash
# Using Makefile (pass N_TREES as a variable)
make run N_TREES=50

# Or directly with uv
uv run python src/main.py --n_trees 50

# Or with standard python
python src/main.py --n_trees 50
```

### Development tasks

```bash
# Format code (using black)
make format

# Lint code (using pylint)
make lint

# Check code (lint + format check)
make check

# Run tests
uv run pytest -v

# Run tests with coverage
uv run coverage run -m pytest tests/
uv run coverage report -m
```

## Features

- **Data preprocessing**: Automatic handling of missing values and feature encoding
- **Model training**: Random Forest classifier with configurable parameters
- **Model evaluation**: Comprehensive metrics including accuracy and confusion matrix
- **Modular design**: Clean separation of data loading, splitting, modeling, and evaluation
- **Testing**: Unit tests with pytest and test coverage
- **CI/CD**: Automated testing and code quality checks with GitHub Actions
- **Docker support**: Containerized deployment with Dockerfile and Dev Containers

## GitHub Actions

This project uses GitHub Actions for continuous integration and deployment:

### Automated Workflows

1. **Tests** (`.github/workflows/tests.yml`)
   - Runs on: push and pull requests to `main` branch
   - Executes: pytest test suite
   - Python version: 3.10

2. **Code Quality** (`.github/workflows/codecheck.yml`)
   - Runs on: push and pull requests to `main` branch
   - Executes: ruff linting and formatting checks
   - Excludes: Jupyter notebooks (`*.ipynb`)

3. **Docker Build** (`.github/workflows/prod.yml`)
   - Runs on: push to `main` or `dev` branches
   - Executes: Builds and pushes Docker image to Docker Hub
   - Requires: `DOCKER_USERNAME` and `DOCKER_PASSWORD` secrets

### Viewing Workflow Status

Check the [Actions tab](https://github.com/alanliyue/esilv-mlops-tp/actions) in GitHub to see workflow runs and their status.

## Requirements

- Python >= 3.10, < 3.13
- `uv` package manager
- See `pyproject.toml` for full dependency list

## License

This is an educational project for teaching purposes. The code is provided for learning and educational use only.
