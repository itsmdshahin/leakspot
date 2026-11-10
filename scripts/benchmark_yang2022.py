"""External benchmark on the labelled test inputs of Yang et al. (ASE 2022).

Usage:
    git clone https://github.com/malusamayo/leakage-analysis /tmp/leakage-analysis
    git -C /tmp/leakage-analysis checkout a7d038bfec6b8ddbe21d87dde54b806aecdd79f7
    python scripts/benchmark_yang2022.py /tmp/leakage-analysis

The labels below are copied from that repository's tests/test_preprocessing.py
and tests/test_overlap.py: a file is a positive if its expected list of leaky
calls is non-empty. Evaluation is at file level: did LeakSpot report any
LS101-LS105 finding in the file? (LS106 is a low-severity advisory and is
excluded.) The original files are not redistributed here.
"""

from __future__ import annotations

import sys
from pathlib import Path

from leakspot.scanner import scan_file

SUITES = {
    "preprocessing": {
        "leak": "test0 test1 test2 titanic0 nb_344873 nb_100841 nb_334422 nb_132929 "
                "nb_194503 nb_362989 nb_205857 nb_471253 nb_303674 nb_276778 nb_387986 "
                "nb_344814 nb_282393 nb_175471 nb_248151".split(),
        "clean": "test3 test4 nb_473437 nb_1020535 nb_277256".split(),
    },
    "overlap": {
        "leak": ("testOversampler yogmoh_news-category "
                 "dktalaicha_credit-card-fraud-detection-using-smote-adasyn "
                 "shahules_tackling-class-imbalance").split(),
        "clean": "testOversampler2 nb_598984".split(),
    },
}


def main(repo: str) -> None:
    base = Path(repo) / "tests" / "inputs"
    rows, tot = [], {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
    for suite, labels in SUITES.items():
        s = {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
        for label, names in labels.items():
            for n in names:
                res = scan_file(base / f"{n}.py")
                rules = sorted({f.rule for f in res.findings if f.rule != "LS106"})
                hit = bool(rules)
                key = {("leak", True): "tp", ("leak", False): "fn",
                       ("clean", True): "fp", ("clean", False): "tn"}[(label, hit)]
                s[key] += 1
                rows.append((suite, label, n, ",".join(rules) or "-", key.upper()))
        for k in tot:
            tot[k] += s[k]
        print(f"{suite:14} TP={s['tp']} FN={s['fn']} FP={s['fp']} TN={s['tn']}  "
              f"recall={s['tp'] / max(1, s['tp'] + s['fn']):.2f}")
    p = tot["tp"] / max(1, tot["tp"] + tot["fp"])
    r = tot["tp"] / max(1, tot["tp"] + tot["fn"])
    print(f"{'all':14} TP={tot['tp']} FN={tot['fn']} FP={tot['fp']} TN={tot['tn']}  "
          f"precision={p:.2f} recall={r:.2f}")
    print()
    for row in rows:
        print("  ".join(f"{c:<12}" if i < 2 else str(c) for i, c in enumerate(row)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/leakage-analysis")
