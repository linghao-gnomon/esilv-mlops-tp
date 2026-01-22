# Titanic Survival Prediction

A machine learning project for predicting passenger survival on the Titanic using Random Forest classification.

## Project Structure

```
.
├── data/
│   └── raw/           # Raw data files
├── notebooks/         # Jupyter notebooks for exploration
├── src/               # Source code
│   ├── data/          # Data utilities
│   │   ├── load.py    # Data loading
│   │   └── split.py   # Data splitting
│   ├── model.py       # Model creation and training
│   ├── evaluation.py  # Model evaluation
│   ├── constants.py   # Project constants
│   └── main.py        # Main entry point
├── Makefile           # Common tasks
└── pyproject.toml     # Project dependencies
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
# Format code
make format

# Lint code
make lint

# Check code (lint + format check)
make check
```

## Features

- **Data preprocessing**: Automatic handling of missing values and feature encoding
- **Model training**: Random Forest classifier with configurable parameters
- **Model evaluation**: Comprehensive metrics including accuracy and confusion matrix
- **Modular design**: Clean separation of data loading, splitting, modeling, and evaluation

## Requirements

- Python >= 3.10, < 3.13
- `uv` package manager
- See `pyproject.toml` for full dependency list

## License

This is an educational project for teaching purposes. The code is provided for learning and educational use only.
