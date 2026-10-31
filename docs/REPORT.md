# Group report

Complete the fields below after the work has actually been done.

## Repository and team

- Repository URL:
- Initial main commit (`git rev-parse initial-main`):
- Final main commit:
- Members: MohammadJavad Norouzi; Aland Arif; Hamed Zahraei Ramazani;
  Wenyuan Huang.
- Four-person allocation: Student 2's pytest and NumPy branches are assigned
  to Hamed and Wenyuan respectively. CI and the final refactor are shared.

## Pull requests in integration order

| Branch | Lead | PR link | Peer approval | CI result |
| --- | --- | --- | --- | --- |
| feature/ci-pipeline | MohammadJavad / team | | | |
| feature/pure-python-ops | Aland | | | |
| feature/pytest-suite | Hamed | | | |
| feature/numpy-vectorization | Wenyuan | | | |
| feature/batched-operations | MohammadJavad | | | |
| feature/modular-refactor | MohammadJavad / team | | | |

## Branch origins

Record how all six branches were created from the tagged initial main
before integration. Link the relevant commit history. Later merge commits
that bring main into a feature branch are expected.

## Implementation

- Pure Python products: describe the two and three nested loops.
- NumPy conversion: describe array storage, `@`, and preserved public methods.
- Batch operations: describe paired lists and stacked array shapes.
- Modular refactor: list the new files and confirm deletion of `core.py`.

## Conflicts actually resolved

| Branch | File(s) that conflicted | Decision and reason | Resolution commit |
| --- | --- | --- | --- |
| | | | |

## Final validation performed by the team

- Local Python version and operating system:
- Command: `python -m pytest -q`; actual result:
- Command: `python -m flake8 src tests`; actual result:
- Coverage command/result, if run:
- Link to final passing GitHub Actions run:
- Tests import the public package and do not reference `core.py`:
- `src/mini_linalg/core.py` is absent:

## Individual contributions

See `docs/contributions/mohammadjavad.md`, `aland.md`, `hamed.md` and
`wenyuan.md`. Each person completes their own record with actual work and
peer reviews.
