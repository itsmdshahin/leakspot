import json
from pathlib import Path

import pytest

from leakspot.scanner import scan_file, scan_source

CORPUS = Path(__file__).parent / "corpus"


def _expected(p: Path) -> str:
    return p.read_text().splitlines()[0].split(":", 1)[1].strip()


@pytest.mark.parametrize("path", sorted((CORPUS / "leaky").glob("*.py")), ids=lambda p: p.name)
def test_leaky_corpus_detected(path):
    rules = {f.rule for f in scan_file(path).findings}
    assert rules == {_expected(path)}


@pytest.mark.parametrize("path", sorted((CORPUS / "clean").glob("*.py")), ids=lambda p: p.name)
def test_clean_corpus_has_no_findings(path):
    assert scan_file(path).findings == []


def test_finding_fields():
    src = (
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.model_selection import train_test_split\n"
        "X, y = load()\n"
        "X = StandardScaler().fit_transform(X)\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
    )
    (f,) = scan_source(src, "demo.py").findings
    assert (f.rule, f.line, f.severity) == ("LS101", 4, "medium")
    assert "line 5" in f.message and f.fix and f.taxonomy.startswith("L1.2")


def test_suppression_all_rules():
    src = ("from sklearn.preprocessing import StandardScaler\n"
           "from sklearn.model_selection import train_test_split\n"
           "X, y = load()\n"
           "X = StandardScaler().fit_transform(X)  # leakspot: ignore\n"
           "X_train, X_test, y_train, y_test = train_test_split(X, y)\n")
    assert scan_source(src).findings == []


def test_syntax_error_is_a_warning_not_a_crash():
    res = scan_source("def broken(:\n")
    assert res.findings == [] and res.warnings


def test_notebook_cells_and_magics(tmp_path):
    cells = [
        "%matplotlib inline\nimport pandas as pd\nfrom sklearn.preprocessing import MinMaxScaler",
        "!pip install something\nfrom sklearn.model_selection import train_test_split\nX, y = load()",
        "X = MinMaxScaler().fit_transform(X)",
        "X_train, X_test, y_train, y_test = train_test_split(X, y)",
    ]
    nb = {"cells": [{"cell_type": "code", "source": c, "metadata": {}, "outputs": []}
                    for c in cells] + [{"cell_type": "markdown", "source": "# notes"}],
          "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    p = tmp_path / "nb.ipynb"
    p.write_text(json.dumps(nb))
    (f,) = scan_file(p).findings
    assert f.rule == "LS101" and f.cell == 2 and f.line == 1
    assert "cell 3" in f.location()


def test_unparseable_notebook_cell_is_skipped(tmp_path):
    nb = {"cells": [{"cell_type": "code", "source": "x = (", "metadata": {}, "outputs": []},
                    {"cell_type": "code", "source": "y = 1", "metadata": {}, "outputs": []}],
          "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    p = tmp_path / "bad.ipynb"
    p.write_text(json.dumps(nb))
    res = scan_file(p)
    assert res.findings == [] and "cell 1" in res.warnings[0]
