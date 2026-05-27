# 🤝 Contributing to PoetryDuel-GPT

Thank you for your interest in contributing! This project welcomes improvements of all sizes.

## Getting Started

```bash
git clone https://github.com/weseegod/poetry-dual-gpt.git
cd poetry-dual-gpt
pip install -r requirements.txt
```

## Development Setup

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v
```

## Project Structure

```
src/            # Core source code
  model.py      # PoetryDuelGPT Transformer
  train.py      # Training loop
  generation.py # Canonical generation pipeline
  tones.py      # Vietnamese tone/rhyme utilities
client/         # FastAPI server + React frontend
evaluate/       # Rule-based and quality evaluation
tests/          # Unit tests
```

## Types of Contributions

### Bug Reports
Open an issue with:
- Environment (OS, Python version, PyTorch version)
- Steps to reproduce
- Expected vs actual behavior
- Error messages and logs

### Code Contributions
1. Fork the repo
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Make changes with clear commit messages
4. Add/update tests
5. Run tests: `python -m pytest tests/ -v`
6. Open a pull request

### Model Improvements
If you'd like to improve the model:
- **Data augmentation** — Adding more Vietnamese poetry data (see `data_service/`)
- **Training hyperparameters** — Experiment with LR schedules, batch sizes, architecture
- **New features** — New control tokens, evaluation metrics, or generation strategies

## Code Style

- Follow PEP 8
- Use type hints for function signatures
- Document public functions with docstrings
- Keep functions focused and small

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_tones.py -v

# Run with coverage
pip install pytest-cov
python -m pytest tests/ --cov=src -v
```

## Adding a New Poetry Rule

1. Define the rule in `src/tones.py` (extraction + validation functions)
2. Add a control token (e.g., `[RULE:X]`) to the training format in preprocessing
3. Update `evaluate/eval_rules.py` with evaluation logic
4. Add tests in `tests/`
5. Retrain the model

## Release Process

1. Update `CHANGELOG.md`
2. Run full evaluation suite: `python evaluate/eval_rules.py && python evaluate/eval_quality.py`
3. Tag the release: `git tag -a vX.Y.Z -m "Release vX.Y.Z"`
4. Push tag: `git push origin vX.Y.Z`

## Questions?

Open a [GitHub Discussion](https://github.com/weseegod/poetry-dual-gpt/discussions) or issue.
