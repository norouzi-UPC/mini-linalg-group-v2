# MiniLinalg

Scientific Programming and Large Scale Computing - group assignment 1.

Team: MohammadJavad Norouzi; Aland Arif; Hamed Zahraei Ramazani; Wenyuan Huang.

## Files

- `src/mini_linalg/vector.py`: Vector.
- `src/mini_linalg/matrix.py`: Matrix and its products.
- `src/mini_linalg/batch.py`: batch products.
- `tests/test_core.py`: tests using the public package.
- `docs/REPORT.md`: short group report.

## Setup and checks

Use Python 3.11 or newer. From the project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m flake8 src tests
```

The feature branches and pull requests on GitHub show each person's work.
