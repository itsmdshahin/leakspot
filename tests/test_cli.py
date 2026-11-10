import json
from pathlib import Path

from click.testing import CliRunner

from leakspot.cli import main

CORPUS = Path(__file__).parent / "corpus"


def test_scan_exit_codes_and_formats():
    r = CliRunner()
    leaky = str(CORPUS / "leaky" / "scale_before_split.py")
    clean = str(CORPUS / "clean" / "pipeline_cv.py")

    res = r.invoke(main, ["scan", leaky])
    assert res.exit_code == 1 and "LS101" in res.output
    assert r.invoke(main, ["scan", clean]).exit_code == 0
    assert r.invoke(main, ["scan", leaky, "--fail-on", "high"]).exit_code == 0
    assert r.invoke(main, ["scan", leaky, "--ignore", "LS101"]).exit_code == 0

    out = json.loads(r.invoke(main, ["scan", leaky, "--format", "json"]).output)
    assert out["findings"][0]["rule"] == "LS101"

    gh = r.invoke(main, ["scan", leaky, "--format", "github"]).output
    assert gh.startswith("::warning file=") and "LS101" in gh


def test_scan_directory():
    res = CliRunner().invoke(main, ["scan", str(CORPUS / "leaky"), "--fail-on", "never"])
    assert res.exit_code == 0 and "19 file(s) scanned" in res.output


def test_audit_command(tmp_path):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=400)})
    df["target"] = (df.x > 0).astype(int)
    df["leak"] = df["target"]
    df.iloc[:300].to_csv(tmp_path / "train.csv", index=False)
    df.iloc[300:].to_csv(tmp_path / "test.csv", index=False)
    res = CliRunner().invoke(main, ["audit", str(tmp_path / "train.csv"),
                                    str(tmp_path / "test.csv"), "-t", "target"])
    assert res.exit_code == 1 and "proxy_feature" in res.output


def test_rules_command():
    out = CliRunner().invoke(main, ["rules"]).output
    assert all(f"LS10{i}" in out for i in range(1, 7))
