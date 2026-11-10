import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split

from leakspot.audit import audit


def _checks(rep):
    return {f.check for f in rep.findings}


@pytest.mark.parametrize("loader", [load_breast_cancer, load_diabetes])
def test_real_clean_datasets_have_no_findings(loader):
    d = loader(as_frame=True).frame
    tr, te = train_test_split(d, test_size=0.3, random_state=0)
    assert audit(tr, te, target="target").findings == []


@pytest.fixture(scope="module")
def leaky():
    rng = np.random.default_rng(0)
    n = 3000
    df = pd.DataFrame({
        "customer_id": rng.integers(0, 600, n),
        "date": pd.date_range("2025-01-01", periods=n, freq="h"),
        "x1": rng.normal(size=n), "x2": rng.normal(size=n),
    })
    df["target"] = (df.x1 + rng.normal(size=n) > 0).astype(int)
    df["refund_amount"] = np.where(df.target == 1, rng.uniform(10, 100, n), 0) \
        + rng.normal(scale=0.5, size=n)
    df["closure_reason"] = np.where(df.target == 1, rng.choice(["a", "b"], n), None)
    df = df.sort_values("target").reset_index(drop=True)
    df["row_number"] = np.arange(n)
    tr, te = train_test_split(df, test_size=0.3, random_state=1)
    te = pd.concat([te, tr.sample(150, random_state=0)])
    return tr, te


def test_planted_leaks_are_all_found(leaky):
    tr, te = leaky
    rep = audit(tr, te, target="target", time_col="date", group_col="customer_id")
    assert {"duplicate_rows", "group_overlap", "temporal_overlap", "proxy_feature",
            "missingness_leak", "id_leak"} <= _checks(rep)
    proxies = {f.details["feature"] for f in rep.findings if f.check == "proxy_feature"}
    assert "refund_amount" in proxies and "x1" not in proxies


def test_clean_time_split_has_no_high_findings():
    rng = np.random.default_rng(1)
    n = 3000
    df = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=n, freq="h"),
                       "customer_id": np.arange(n) // 3,
                       "x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    df["target"] = (df.x1 + rng.normal(size=n) > 0).astype(int)
    cut = int(n * 0.7)
    rep = audit(df.iloc[:cut], df.iloc[cut:], target="target", time_col="date",
                group_col="customer_id")
    assert not [f for f in rep.findings if f.severity == "high"]


def test_regression_target_and_text_report():
    d = load_diabetes(as_frame=True).frame
    d["leak"] = d["target"] * 2 + 1
    tr, te = train_test_split(d, test_size=0.3, random_state=0)
    rep = audit(tr, te, target="target")
    assert rep.task == "regression"
    assert any(f.check == "proxy_feature" and f.details["feature"] == "leak"
               for f in rep.findings)
    assert "proxy_feature" in rep.to_text()
