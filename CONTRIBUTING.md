# Contributing to LeakSpot

1. `pip install -e ".[dev]"` and run `pytest`.
2. **Found a missed leak or a false alarm?** Open an issue with the smallest
   snippet that shows it. Even better, add it to `tests/corpus/leaky/` or
   `tests/corpus/clean/` with an `# expect: LSxxx` (or `# expect: none`) first
   line and open a pull request.
3. New rules need: an entry in `rules.py` with a fix and a taxonomy code, at
   least two leaky and two clean corpus cases, and an updated rules table in
   the README.

Please report benchmark changes honestly. If a change was motivated by a
benchmark miss, say so in the pull request and in `docs/BENCHMARK.md`.
