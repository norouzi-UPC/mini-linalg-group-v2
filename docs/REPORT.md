# Group report

## Repository and team

- Repository URL: https://github.com/norouzi-UPC/mini-linalg-group-v2
- Initial main commit: To be completed.
- Final main commit: To be completed.
- Members: MohammadJavad Norouzi; Aland Arif; Hamed Zahraei Ramazani;
  Wenyuan Huang.

## Pull requests in integration order

| Branch | Lead | PR link | Peer approval | CI result |
| --- | --- | --- | --- | --- |
| feature/ci-pipeline | MohammadJavad / team | [#1](https://github.com/norouzi-UPC/mini-linalg-group-v2/pull/1) | Aland approved | Passed |
| feature/pure-python-ops | Aland | To be completed | To be completed | To be completed |
| feature/pytest-suite | Hamed | To be completed | To be completed | To be completed |
| feature/numpy-vectorization | Wenyuan | [#2](https://github.com/norouzi-UPC/mini-linalg-group-v2/pull/2) | Changes requested | To be completed |
| feature/batched-operations | MohammadJavad | [#3](https://github.com/norouzi-UPC/mini-linalg-group-v2/pull/3) | Pending | To be completed |
| feature/modular-refactor | MohammadJavad / team | To be completed | To be completed | To be completed |

## Notes

- CI: I added pytest and flake8 checks. Aland approved [PR #1](https://github.com/norouzi-UPC/mini-linalg-group-v2/pull/1), and it was merged.
- NumPy review: I reviewed Wenyuan's [PR #2](https://github.com/norouzi-UPC/mini-linalg-group-v2/pull/2) and asked for some fixes. We are waiting for her update.
- Batches: I added batch matrix-vector and matrix-matrix products in [PR #3](https://github.com/norouzi-UPC/mini-linalg-group-v2/pull/3). Locally, 19 tests passed and flake8 passed. The PR is still open.

## Branch origins

To be completed.

## Implementation

- Pure Python products: To be completed.
- NumPy conversion: To be completed.
- Batch operations: Added products for matching matrix-vector and matrix-matrix pairs.
- Modular refactor: To be completed.

## Conflicts actually resolved

| Branch | File(s) that conflicted | Decision and reason | Resolution commit |
| --- | --- | --- | --- |
| To be completed | To be completed | To be completed | To be completed |

## Final validation performed by the team

- Local Python version and operating system: To be completed.
- `python -m pytest -q`: To be completed.
- `python -m flake8 src tests`: To be completed.
- Coverage command/result, if run: To be completed.
- Link to final passing GitHub Actions run: To be completed.
- Tests import the public package and do not reference `core.py`: To be completed.
- `src/mini_linalg/core.py` is absent: To be completed.

## Individual contributions

- `docs/contributions/mohammadjavad.md`
- `docs/contributions/aland.md`
- `docs/contributions/hamed.md`
- `docs/contributions/wenyuan.md`
