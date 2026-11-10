"""Data-level leakage audit for a train/test pair.

Static code analysis cannot see the data. This module checks the actual train
and test tables for the leakage patterns that only show up in the rows:

=====================  ==================================================  ==========
Check                  What it looks for                                   Taxonomy*
=====================  ==================================================  ==========
duplicate_rows         Test rows whose features also appear in train       L1.4
group_overlap          Same entity (customer, patient, ...) in both sets   L3.2
temporal_overlap       Test rows dated before the end of training data     L3.1
proxy_feature          One feature alone predicts the target almost        L2
                       perfectly ("too good to be true")
missingness_leak       Whether a value is missing predicts the target      L2
id_leak                An ID / row-number column predicts the target       L2 / L3.3
split_shift            A model can tell train rows from test rows          L3.x
                       (adversarial validation), with the features most
                       responsible
=====================  ==================================================  ==========

\\* Kapoor & Narayanan (2023), Patterns 4(9).

Every check returns evidence, not a verdict: a strong proxy feature might be
legitimate, and some split shift is expected with a time-based split. The
thresholds are configurable and documented in :class:`AuditConfig`.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


@dataclass
class AuditConfig:
    duplicate_warn: float = 0.01      # share of test rows duplicated in train
    proxy_auc: float = 0.98           # single-feature AUC (classification)
    proxy_r2: float = 0.95            # single-feature R² (regression)
    missing_auc: float = 0.80         # AUC of an is-missing indicator
    id_auc: float = 0.70              # AUC of an ID-like column
    shift_auc: float = 0.75           # adversarial validation AUC
    max_rows: int = 50_000            # rows sampled per set for model-based checks
    seed: int = 0


@dataclass
class AuditFinding:
    check: str
    severity: str
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class AuditReport:
    task: str
    n_train: int
    n_test: int
    findings: list[AuditFinding] = field(default_factory=list)
    checks_run: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_text(self) -> str:
        lines = [f"LeakSpot data audit ({self.task}; train={self.n_train:,} rows, "
                 f"test={self.n_test:,} rows)",
                 f"Checks run: {', '.join(self.checks_run)}"]
        if not self.findings:
            lines.append("No leakage signals found.")
        for f in self.findings:
            lines.append(f"[{f.severity.upper():6}] {f.check}: {f.message}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- helpers
def _task(y: pd.Series) -> str:
    if y.dtype == object or str(y.dtype) in ("category", "bool") or y.nunique() <= 20:
        return "classification"
    return "regression"


def _encode(s: pd.Series) -> np.ndarray:
    """Numeric version of a column for single-feature trees (NaN -> sentinel)."""
    if pd.api.types.is_datetime64_any_dtype(s):
        v = s.astype("int64").astype(float).to_numpy().copy()
        v[s.isna().to_numpy()] = np.nan
    elif pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        v = s.astype(float).to_numpy().copy()
    else:
        v = s.astype("category").cat.codes.astype(float).to_numpy().copy()
        v[v < 0] = np.nan
    v = np.asarray(v, dtype=float)
    fill = np.nanmin(v) - 1 if np.isfinite(v).any() else -1
    return np.where(np.isnan(v), fill, v).reshape(-1, 1)


def _single_feature_score(x: np.ndarray, y: np.ndarray, task: str, seed: int) -> float:
    """Out-of-fold score of a depth-4 tree using one feature."""
    if task == "classification":
        classes, yi = np.unique(y, return_inverse=True)
        if len(classes) < 2 or np.bincount(yi).min() < 3:
            return float("nan")
        cv = StratifiedKFold(3, shuffle=True, random_state=seed)
        p = cross_val_predict(DecisionTreeClassifier(max_depth=4, min_samples_leaf=5,
                                                     random_state=seed),
                              x, yi, cv=cv, method="predict_proba")
        if len(classes) == 2:
            return float(roc_auc_score(yi, p[:, 1]))
        return float(roc_auc_score(yi, p, multi_class="ovr"))
    cv = KFold(3, shuffle=True, random_state=seed)
    p = cross_val_predict(DecisionTreeRegressor(max_depth=4, min_samples_leaf=5,
                                                random_state=seed), x, y, cv=cv)
    return float(r2_score(y, p))


def _sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return df.sample(n, random_state=seed) if len(df) > n else df


# --------------------------------------------------------------------------- audit
def audit(train: pd.DataFrame, test: pd.DataFrame, target: str,
          time_col: str | None = None, group_col: str | None = None,
          id_cols: list[str] | None = None, config: AuditConfig | None = None
          ) -> AuditReport:
    """Audit a train/test pair for data-level leakage.

    Parameters
    ----------
    train, test : DataFrames with the same columns. ``test`` may lack ``target``.
    target      : name of the label column.
    time_col    : timestamp column, if the split is meant to be by time.
    group_col   : entity column (customer, patient...) that must not cross the split.
    id_cols     : columns to treat as identifiers. If None, LeakSpot guesses
                  (unique integer or string columns, or names ending in "id").
    """
    cfg = config or AuditConfig()
    if target not in train.columns:
        raise ValueError(f"target column {target!r} not in train")
    task = _task(train[target])
    rep = AuditReport(task=task, n_train=len(train), n_test=len(test))
    special = {target, time_col, group_col} - {None}
    if id_cols is None:
        id_cols = [c for c in train.columns if c not in special and (
            c.lower() in {"id", "index", "row_id", "uuid"} or re.search(r"(_id|Id|ID)$", c)
            or (train[c].nunique() >= 0.98 * len(train)
                and (pd.api.types.is_integer_dtype(train[c]) or train[c].dtype == object)))]
    features = [c for c in train.columns if c not in special and c not in id_cols
                and c in test.columns]

    # ---- duplicate rows across the split
    rep.checks_run.append("duplicate_rows")
    if features:
        h_tr = set(pd.util.hash_pandas_object(train[features], index=False))
        h_te = pd.util.hash_pandas_object(test[features], index=False)
        dup = float(h_te.isin(h_tr).mean())
        if dup > cfg.duplicate_warn:
            rep.findings.append(AuditFinding(
                "duplicate_rows", "high" if dup > 0.05 else "medium",
                f"{dup:.1%} of test rows have an identical feature row in train "
                f"({int(h_te.isin(h_tr).sum()):,} rows). The model may be memorising "
                "rather than generalising.", {"share": round(dup, 4)}))

    # ---- entity overlap
    if group_col:
        rep.checks_run.append("group_overlap")
        g_tr, g_te = set(train[group_col].dropna()), set(test[group_col].dropna())
        both = g_tr & g_te
        if both:
            share_rows = float(test[group_col].isin(both).mean())
            rep.findings.append(AuditFinding(
                "group_overlap", "high",
                f"{len(both):,} {group_col} values appear in both train and test "
                f"({share_rows:.1%} of test rows). Use a grouped split (GroupKFold / "
                "GroupShuffleSplit) if you need to generalise to new entities.",
                {"overlapping_groups": len(both), "test_row_share": round(share_rows, 4)}))

    # ---- temporal overlap
    if time_col:
        rep.checks_run.append("temporal_overlap")
        t_tr = pd.to_datetime(train[time_col])
        t_te = pd.to_datetime(test[time_col])
        cut = t_tr.max()
        early = float((t_te < cut).mean())
        if early > 0:
            rep.findings.append(AuditFinding(
                "temporal_overlap", "high" if early > 0.05 else "medium",
                f"{early:.1%} of test rows are dated before the last training row "
                f"({cut}). The model is trained partly on the future relative to those "
                "rows.", {"share_before_train_end": round(early, 4), "train_end": str(cut)}))

    # ---- single-feature checks on (a sample of) train
    tr = _sample(train, cfg.max_rows, cfg.seed)
    y = tr[target].to_numpy()
    if task == "regression":
        y = y.astype(float)

    rep.checks_run.append("proxy_feature")
    thr = cfg.proxy_auc if task == "classification" else cfg.proxy_r2
    metric = "AUC" if task == "classification" else "R²"
    scores = {}
    for c in features:
        sc = _single_feature_score(_encode(tr[c]), y, task, cfg.seed)
        scores[c] = sc
        if np.isfinite(sc) and sc >= thr:
            rep.findings.append(AuditFinding(
                "proxy_feature", "high",
                f"`{c}` alone predicts `{target}` with {metric} {sc:.3f}. Check that it "
                "is known at prediction time and is not derived from the target.",
                {"feature": c, "score": round(sc, 4), "metric": metric}))

    rep.checks_run.append("missingness_leak")
    if task == "classification":
        for c in features:
            miss = tr[c].isna()
            if 0.01 < miss.mean() < 0.99:
                sc = _single_feature_score(miss.to_numpy(float).reshape(-1, 1), y, task, cfg.seed)
                if np.isfinite(sc) and sc >= cfg.missing_auc:
                    rep.findings.append(AuditFinding(
                        "missingness_leak", "medium",
                        f"Whether `{c}` is missing predicts `{target}` (AUC {sc:.3f}). "
                        "Missingness is often recorded *after* the outcome (e.g. a field "
                        "only filled in for defaulted loans).",
                        {"feature": c, "auc": round(sc, 4)}))

    if id_cols:
        rep.checks_run.append("id_leak")
        for c in id_cols:
            if c not in tr.columns:
                continue
            sc = _single_feature_score(_encode(tr[c]), y, task, cfg.seed)
            lim = cfg.id_auc if task == "classification" else 0.3
            if np.isfinite(sc) and sc >= lim:
                rep.findings.append(AuditFinding(
                    "id_leak", "medium",
                    f"Identifier column `{c}` predicts `{target}` ({metric} {sc:.3f}). "
                    "The data was probably sorted or collected by outcome; drop the "
                    "column and check how rows were ordered and split.",
                    {"column": c, "score": round(sc, 4)}))

    # ---- adversarial validation: can a model tell train from test?
    rep.checks_run.append("split_shift")
    if features:
        a = _sample(train[features], cfg.max_rows, cfg.seed)
        b = _sample(test[features], cfg.max_rows, cfg.seed)
        both = pd.concat([a, b], ignore_index=True)
        lab = np.r_[np.zeros(len(a)), np.ones(len(b))]
        per = {}
        for c in features:
            per[c] = _single_feature_score(_encode(both[c]), lab, "classification", cfg.seed)
        top = sorted(((v, k) for k, v in per.items() if np.isfinite(v)), reverse=True)[:5]
        if top and top[0][0] >= cfg.shift_auc:
            msg_feats = ", ".join(f"`{k}` (AUC {v:.2f})" for v, k in top if v >= 0.6)
            sev = "low" if time_col else "medium"
            rep.findings.append(AuditFinding(
                "split_shift", sev,
                f"Train and test rows are easy to tell apart, mostly through {msg_feats}. "
                + ("Expected for a time-based split; check these are not time proxies "
                   "that leak the future." if time_col else
                   "For a random split this is suspicious: check how the split was made."),
                {"top_features": {k: round(v, 4) for v, k in top}}))

    order = {"high": 0, "medium": 1, "low": 2}
    rep.findings.sort(key=lambda f: order[f.severity])
    return rep
