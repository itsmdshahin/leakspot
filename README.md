# LeakSpot

[![CI](https://github.com/itsmdshahin/leakspot/actions/workflows/ci.yml/badge.svg)](https://github.com/itsmdshahin/leakspot/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](docs/BENCHMARK.md)

**Find data leakage in machine-learning code and data, before it inflates your results.**

Data leakage happens when information from the test set reaches the model
during training. The model then looks better offline than it really is. It is
one of the most common and most damaging mistakes in applied ML:

- A static analysis of over 100,000 public notebooks found leakage in nearly
  30% of them (Yang et al., ASE 2022).
- A survey of reviews found leakage affecting at least 294 papers across 17
  scientific fields (Kapoor & Narayanan, *Patterns*, 2023).

LeakSpot catches leakage in two complementary ways:

| | `leakspot scan` | `leakspot audit` |
|---|---|---|
| Looks at | Your **code** (`.py` and `.ipynb`) | Your **data** (train and test tables) |
| Finds | Preprocessing before the split, fitting on test data, SMOTE before the split, preprocessing outside CV, target encoding before the split, shuffled time-series splits | Duplicate rows across the split, the same entity in both sets, test rows dated before training ends, "too good to be true" features, missingness leaks, ID leaks, train/test shift |
| Runs | In milliseconds, no execution needed | Seconds, on a sample of the data |
| Use in | Pre-commit, CI (GitHub annotations), code review, teaching | Before trusting any evaluation result |

## Contents

- [Install](#install)
- [Quick start](#quick-start)
- [Rules (static scan)](#rules-static-scan)
- [Data audit checks](#data-audit-checks)
- [Use in CI and pre-commit](#use-in-ci-and-pre-commit)
- [How accurate is it?](#how-accurate-is-it)
- [How it relates to existing tools](#how-it-relates-to-existing-tools)
- [Limitations](#limitations)
- [Python API](#python-api)
- [Roadmap](#roadmap)
- [Contributing](#contributing) · [Citing](#citing-leakspot) · [Licence](#licence)

## Install

```bash
pip install git+https://github.com/itsmdshahin/leakspot.git
```

Requires Python 3.10+. Dependencies: click, numpy, pandas, scikit-learn.

## Quick start

```bash
leakspot scan my_notebook.ipynb src/          # files or whole folders
leakspot audit train.csv test.csv -t target --time-col date --group-col customer_id
leakspot rules                                # list the rules
```

Example: `leakspot scan` on a typical notebook mistake:

```python
X = StandardScaler().fit_transform(X)                              # line 4
X_train, X_test, y_train, y_test = train_test_split(X, y)          # line 5
```

```
model.py:4: LS101 [medium] preprocessing-before-split
    `StandardScaler.fit_transform(X)` learns from the full dataset before the train/test split on line 5.
    fix: Split first and fit preprocessing on the training set only, or put the step inside a scikit-learn Pipeline.
```

Example: `leakspot audit` on a dataset with planted problems:

```
[HIGH  ] group_overlap: 658 customer_id values appear in both train and test (98.2% of test rows). ...
[HIGH  ] temporal_overlap: 99.8% of test rows are dated before the last training row ...
[HIGH  ] proxy_feature: `refund_amount` alone predicts `target` with AUC 1.000. Check that it is known at prediction time ...
[MEDIUM] missingness_leak: Whether `closure_reason` is missing predicts `target` (AUC 1.000). ...
[MEDIUM] id_leak: Identifier column `row_number` predicts `target` (AUC 0.999). ...
```

## Rules (static scan)

Taxonomy codes refer to Kapoor & Narayanan (2023).

| Rule | Name | Example | Severity | Taxonomy |
|---|---|---|---|---|
| LS101 | preprocessing-before-split | `StandardScaler().fit_transform(X)`, `SelectKBest(...).fit(X, y)`, `df[c].fillna(df[c].mean())` before `train_test_split` | medium (high for feature selection, low for encoders) | L1.2, L1.3 |
| LS102 | resampling-before-split | `SMOTE().fit_resample(X, y)` before the split | high | L1.2 |
| LS103 | fit-on-test-data | `scaler.fit_transform(X_test)`, `model.fit(pd.concat([X_train, X_test]))`, early stopping with `eval_set=[(X_test, y_test)]` | high | L1.1, L1.2 |
| LS104 | preprocessing-outside-cv | `X = SelectKBest().fit_transform(X, y)` then `cross_val_score(model, X, y)` | high for selection/resampling/target encoding, low otherwise | L1.2 |
| LS105 | target-encoding-before-split | `df.groupby("region")["defaulted"].transform("mean")` before the split | high | L1.2, L2 |
| LS106 | shuffled-split-on-time-data | `train_test_split` without `shuffle=False` on data with dates | low (advisory) | L3.1 |

To silence a finding you have reviewed, add `# leakspot: ignore` or
`# leakspot: ignore[LS101]` to the line.

## Data audit checks

| Check | What it detects | Taxonomy |
|---|---|---|
| `duplicate_rows` | Test rows whose features also appear in train | L1.4 |
| `group_overlap` | The same entity (customer, patient, ...) in both sets (`--group-col`) | L3.2 |
| `temporal_overlap` | Test rows dated before the last training row (`--time-col`) | L3.1 |
| `proxy_feature` | One feature alone predicts the target almost perfectly (out-of-fold AUC ≥ 0.98 or R² ≥ 0.95) | L2 |
| `missingness_leak` | Whether a value is missing predicts the target (AUC ≥ 0.80) | L2 |
| `id_leak` | An identifier or row-number column predicts the target | L2, L3.3 |
| `split_shift` | Single-feature adversarial validation: which features tell train and test apart | L3 |

The audit reports **evidence, not verdicts**. A strong single feature may be
perfectly legitimate, and some train/test shift is expected with a time-based
split. All thresholds are configurable in `leakspot.AuditConfig`.

## Use in CI and pre-commit

The exit code is 1 when there is a finding at or above `--fail-on`
(default `medium` for `scan`, `high` for `audit`).

**GitHub Actions**, with inline annotations on the pull request:

```yaml
- run: pip install git+https://github.com/itsmdshahin/leakspot.git
- run: leakspot scan . --format github
```

**pre-commit:**

```yaml
repos:
  - repo: https://github.com/itsmdshahin/leakspot
    rev: v0.1.0
    hooks:
      - id: leakspot
```

`--format json` gives machine-readable output for other tools.

## How accurate is it?

**External benchmark.** I used the labelled test inputs published by Yang et
al. (2022), which mix small hand-written cases with real Kaggle notebooks (30
labelled files). Each file is scored on whether LeakSpot reports any LS101 to
LS105 finding.

| | Precision | Recall | TP / FN / FP / TN |
|---|---|---|---|
| First run, before any tuning | 0.89 | 0.74 | 17 / 6 / 2 / 5 |
| v0.1.0 (after the fixes listed below) | 0.91 | 0.87 | 20 / 3 / 2 / 5 |

Please read these numbers with care:

- **The benchmark is small**: 30 files, only 7 of them negative.
- **The v0.1.0 numbers are optimistic.** Four general fixes were written after
  looking at the first run's misses: `np.mean()`-style statistics,
  statistics computed on the test set, test sets loaded from disk, and
  full-data statistics written into a split. A fair estimate needs a
  held-out set, which is the top roadmap item.
- **Both "false positives" are, in my reading, real bugs.** `nb_598984`
  calls `StandardScaler().fit_transform(X_test)`. `test4` unpacks
  `train_test_split` outputs in the wrong order, so test rows really are
  passed to `fit`. They are counted as errors anyway, because that is what
  the labels say.
- **Not directly comparable** with Yang et al.'s reported 92.9% accuracy,
  which was measured differently and on a different sample.

The full per-file breakdown is in [docs/BENCHMARK.md](docs/BENCHMARK.md).

**Regression corpus.** `tests/corpus/` holds 19 leaky and 12 clean snippets,
including tricky clean cases: Pipelines inside CV, SMOTE on the training set
only, early stopping on a validation split, and refitting on all data for
deployment. These are written by the author, so passing them is a regression
test, not evidence of accuracy.

**Data audit.** It finds all seven planted leak types in a synthetic test, and
reports nothing on random splits of three real scikit-learn datasets (breast
cancer, diabetes, wine). This is a sanity check, not a benchmark.

## How it relates to existing tools

LeakSpot builds on published work rather than replacing it:

- **Yang et al. (ASE 2022)** created the static-analysis approach and the
  large-scale study ([leakage-analysis](https://github.com/malusamayo/leakage-analysis)).
  Their tool uses a Datalog-based data-flow analysis and runs in Docker, and
  their paper describes it as aimed at large-scale empirical studies.
- **LeakageDetector** (AlOmar et al., 2025) wraps that analysis in PyCharm and
  VS Code extensions with quick fixes.
- **LeakSpot's focus** is everyday use: `pip install`, pure Python, no Docker,
  notebook-native, CI and pre-commit friendly with GitHub annotations. It
  also adds a **data audit**, because several important leak types (duplicates,
  entity overlap, temporal overlap, proxy features) cannot be seen in code at
  all. Findings are mapped to the Kapoor & Narayanan taxonomy.

The trade-off is that LeakSpot's analysis is simpler than a full Datalog
data-flow analysis. See Limitations.

## Limitations

- **Approximate data flow.** Function and loop bodies are read once, in
  order. Notebooks are assumed to run top to bottom. Leakage hidden inside a
  user-defined function that is applied later (e.g. via `df.apply(f)`) is
  missed.
- **Lineage over-approximates.** Any expression that mentions a dataset is
  treated as derived from it. This can cause false positives when unrelated
  data is combined.
- **Manual splits are recognised by name** (`*test*`, `*holdout*`, `*_te`).
  Splits written as boolean masks with neutral names are not detected.
- **Not tracked yet**: function-style resampling (`sklearn.utils.resample`),
  PyTorch/TensorFlow data pipelines, and "multi-test" leakage (choosing models
  by repeated test-set evaluation).
- **The data audit's adversarial validation uses single features**, so it
  can miss shifts that only appear in feature combinations.

## Python API

```python
from leakspot import scan_file, audit

for f in scan_file("analysis.ipynb").findings:
    print(f.location(), f.rule, f.message)

report = audit(train_df, test_df, target="label", time_col="date", group_col="user_id")
print(report.to_text())
```

## Roadmap

- [ ] **Held-out evaluation** on a fresh set of labelled public notebooks
- [ ] Multi-test leakage (repeated test-set evaluation for model selection)
- [ ] Inter-procedural analysis for user-defined preprocessing functions
- [ ] PyTorch / TensorFlow / Keras data-pipeline patterns
- [ ] Multivariate adversarial validation in the data audit
- [ ] Auto-fix suggestions (rewrite into a `Pipeline`)

## Contributing

Bug reports with a minimal code snippet are the most valuable contribution:
each one becomes a new corpus case. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Citing LeakSpot

```
Rahman, M. S. (2026). LeakSpot: finding data leakage in machine-learning code
and data (Version 0.1.0) [Computer software]. https://github.com/itsmdshahin/leakspot
```

Please also cite the work it builds on:

- Yang, C., Brower-Sinning, R. A., Lewis, G. A., & Kästner, C. (2022). *Data
  Leakage in Notebooks: Static Detection and Better Processes.* ASE '22.
  https://doi.org/10.1145/3551349.3556918
- Kapoor, S., & Narayanan, A. (2023). *Leakage and the reproducibility crisis
  in machine-learning-based science.* Patterns, 4(9), 100804.
  https://doi.org/10.1016/j.patter.2023.100804

## Licence

MIT © Md Shahinur Rahman
