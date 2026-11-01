# Aland Arif

- Branch and pull request: `feature/pure-python-ops` https://github.com/norouzi-UPC/mini-linalg-group-v2/pull/4
- Changes made: added matrix-vector (`mat_vec`) and matrix-matrix
  (`mat_mat`) products with plain Python loops in `src/mini_linalg/core.py`,
  plus indexing, length and dimension checks they need. Added 4 tests in
  `tests/test_core.py`. Code taken from the team task pack and checked by me.
- Tests run and results: 31 Oct 2026, `pytest -q` gave 8 passed;
  `flake8 src tests` gave no warnings.
- Peer review given (PR link): reviewed and approved the CI PR
  https://github.com/norouzi-UPC/mini-linalg-group-v2/pull/1 and the batch PR
  https://github.com/norouzi-UPC/mini-linalg-group-v2/pull/3
  (re-checked after its conflict merge).
- Conflict resolved, if applicable:
