# Benchmark details

## External: Yang et al. (ASE 2022) labelled test inputs

**Source:** the `tests/inputs` files and the expected results in
`tests/test_preprocessing.py` and `tests/test_overlap.py` of
[malusamayo/leakage-analysis](https://github.com/malusamayo/leakage-analysis)
(MIT licence), at commit `a7d038b`. The files are not redistributed here.
Reproduce with:

```bash
git clone https://github.com/malusamayo/leakage-analysis /tmp/leakage-analysis
git -C /tmp/leakage-analysis checkout a7d038bfec6b8ddbe21d87dde54b806aecdd79f7
python scripts/benchmark_yang2022.py /tmp/leakage-analysis
```

**Protocol:** file level. A file counts as positive if its expected list of
leaky calls is non-empty. LeakSpot "detects" a file if it reports any LS101 to
LS105 finding. LS106 is advisory and is excluded.

**History, so that the tuning is visible:**

| Version | TP | FN | FP | TN | Precision | Recall |
|---|---|---|---|---|---|---|
| First run (no tuning) | 17 | 6 | 2 | 5 | 0.89 | 0.74 |
| + `np.mean()`-style statistics, statistics on test data, test files loaded from disk | 19 | 4 | 2 | 5 | 0.90 | 0.83 |
| + full-data statistic written into a split (v0.1.0) | 20 | 3 | 2 | 5 | 0.91 | 0.87 |

**Remaining misses (v0.1.0):**

- `nb_362989`: a scaler is fitted inside a user-defined function that is
  applied later. This needs inter-procedural analysis.
- `nb_344814`: the *label* is defined using the mean of the full dataset
  (`x > wdf['ann_sal'].mean()`). This is not modelled.
- `shahules_tackling-class-imbalance`: resampling uses the function
  `sklearn.utils.resample`, which is not tracked yet.

**"False positives" (v0.1.0), which I believe are real bugs:**

- `nb_598984` (overlap suite, labelled clean): `StandardScaler().fit_transform(X_test)`.
  This is an LS103 issue, which the overlap suite does not label.
- `test4` (labelled clean): `X_train_0, y_train, X_test_0, y_test = train_test_split(data, y)`
  unpacks outputs in the wrong order, so `y_train` actually holds test rows of
  `data`, and it is passed to `model.fit`.

## Raw output (v0.1.0)

```
preprocessing  TP=17 FN=2 FP=1 TN=4  recall=0.89
overlap        TP=3 FN=1 FP=1 TN=1  recall=0.75
all            TP=20 FN=3 FP=2 TN=5  precision=0.91 recall=0.87

preprocessing  leak          test0  LS101  TP
preprocessing  leak          test1  LS103  TP
preprocessing  leak          test2  LS101  TP
preprocessing  leak          titanic0  LS103  TP
preprocessing  leak          nb_344873  LS103  TP
preprocessing  leak          nb_100841  LS101  TP
preprocessing  leak          nb_334422  LS101  TP
preprocessing  leak          nb_132929  LS101  TP
preprocessing  leak          nb_194503  LS101,LS103  TP
preprocessing  leak          nb_362989  -  FN
preprocessing  leak          nb_205857  LS101  TP
preprocessing  leak          nb_471253  LS101  TP
preprocessing  leak          nb_303674  LS101  TP
preprocessing  leak          nb_276778  LS104  TP
preprocessing  leak          nb_387986  LS101  TP
preprocessing  leak          nb_344814  -  FN
preprocessing  leak          nb_282393  LS101  TP
preprocessing  leak          nb_175471  LS101  TP
preprocessing  leak          nb_248151  LS101  TP
preprocessing  clean         test3  -  TN
preprocessing  clean         test4  LS103  FP
preprocessing  clean         nb_473437  -  TN
preprocessing  clean         nb_1020535  -  TN
preprocessing  clean         nb_277256  -  TN
overlap       leak          testOversampler  LS102  TP
overlap       leak          yogmoh_news-category  LS103  TP
overlap       leak          dktalaicha_credit-card-fraud-detection-using-smote-adasyn  LS101,LS102,LS104  TP
overlap       leak          shahules_tackling-class-imbalance  -  FN
overlap       clean         testOversampler2  -  TN
overlap       clean         nb_598984  LS103  FP
```
