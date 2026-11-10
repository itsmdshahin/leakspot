"""Static detection of data leakage in Python scripts and Jupyter notebooks.

How it works
------------
The scanner walks the program's statements in source order (notebook cells in
file order) and keeps a light-weight *lineage* for every variable: the set of
data sources it was derived from, plus tags such as ``TEST#12`` (produced by a
split on line 12) or ``PRE:select#7`` (output of a feature selector fitted on
line 7). It records four kinds of events: train/test splits, ``fit``-style
calls, cross-validation calls and target-aggregation expressions. The rules in
:mod:`leakspot.rules` are then checked against those events.

Known limitations (by design, to stay fast and dependency-free)
---------------------------------------------------------------
* Flow-insensitive inside functions and loops: bodies are read once, in order.
* Notebooks are assumed to run top-to-bottom.
* Lineage passes through any expression that mentions a variable, which can
  over-approximate (false positives) when unrelated data is combined.
* Manual splits are recognised by variable names containing ``test``,
  ``holdout`` or ``_te``.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .rules import (STAT_FUNCS, CV_FUNCS, FIT_METHODS, HIGH_RISK_KINDS, IGNORED_TRANSFORMERS,
                    PIPELINES, RULES, SEARCHES, SEVERITY_ORDER, STAT_METHODS,
                    TRANSFORMERS)

_TEST_NAME = re.compile(r"(test|holdout|(^|_)te(_|$|\d))", re.I)
_TRAIN_NAME = re.compile(r"(train|(^|_)tr(_|$|\d))", re.I)
_TIME_HINT = re.compile(r"to_datetime|parse_dates|DatetimeIndex|\.dt\.|datetime64|"
                        r"resample\(['\"]|date_range")
_DATA_CTORS = {"DataFrame", "Series", "Index", "Categorical", "DMatrix", "Dataset",
               "Pool", "Tensor", "TensorDataset", "Path"}
_SUPPRESS = re.compile(r"#\s*leakspot:\s*ignore(?:\[([A-Z0-9,\s]+)\])?", re.I)


@dataclass
class Finding:
    rule: str
    name: str
    severity: str
    path: str
    line: int
    message: str
    fix: str
    taxonomy: str
    cell: int | None = None

    def location(self) -> str:
        if self.cell is not None:
            return f"{self.path}:cell {self.cell + 1}, line {self.line}"
        return f"{self.path}:{self.line}"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScanResult:
    path: str
    findings: list[Finding] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- utils
def _callee(node: ast.AST) -> str | None:
    """Name of a call's function: foo(...) -> foo, a.b.foo(...) -> foo."""
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _src(node: ast.AST, limit: int = 40) -> str:
    try:
        s = ast.unparse(node)
    except Exception:  # pragma: no cover
        s = "<expr>"
    return s if len(s) <= limit else s[: limit - 3] + "..."


def _arg(call: ast.Call, pos: int, *kw: str) -> ast.AST | None:
    for k in call.keywords:
        if k.arg in kw:
            return k.value
    return call.args[pos] if len(call.args) > pos else None


def _target_names(t: ast.AST) -> list[str]:
    if isinstance(t, ast.Name):
        return [t.id]
    if isinstance(t, (ast.Tuple, ast.List)):
        return [n for e in t.elts for n in _target_names(e)]
    if isinstance(t, ast.Starred):
        return _target_names(t.value)
    return []


def _base_name(t: ast.AST) -> str | None:
    """df['a'] / df.loc[...] / df.a  ->  df"""
    while isinstance(t, (ast.Subscript, ast.Attribute)):
        t = t.value
    return t.id if isinstance(t, ast.Name) else None


# --------------------------------------------------------------------------- source
def _load_source(path: Path) -> tuple[str, list[tuple[int | None, int]], list[str]]:
    """Return (python source, line map, warnings). Line map: global line -> (cell, line)."""
    text = path.read_text(encoding="utf-8")
    if path.suffix != ".ipynb":
        n = text.count("\n") + 1
        return text, [(None, i + 1) for i in range(n)], []

    nb = json.loads(text)
    lines: list[str] = []
    lmap: list[tuple[int | None, int]] = []
    warnings: list[str] = []
    for ci, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        src = "".join(src) if isinstance(src, list) else src
        cell_lines = []
        for ln in src.splitlines():
            s = ln.lstrip()
            # IPython magics and shell escapes are not Python; keep line numbers.
            cell_lines.append("" if s.startswith(("%", "!", "?")) or s.endswith("?") else ln)
        try:
            ast.parse("\n".join(cell_lines))
        except SyntaxError:
            warnings.append(f"cell {ci + 1}: could not be parsed and was skipped")
            cell_lines = [""] * len(cell_lines)
        for i, ln in enumerate(cell_lines):
            lines.append(ln)
            lmap.append((ci, i + 1))
    return "\n".join(lines), lmap, warnings


# --------------------------------------------------------------------------- analysis
@dataclass
class _Split:
    line: int
    inputs: frozenset
    manual: bool = False


@dataclass
class _Fit:
    line: int
    obj: str | None
    cls: str | None
    kind: str           # transformer kind, "model", "search", "pipeline"
    method: str
    arg_text: str
    lineage: frozenset


class _Analyzer:
    def __init__(self, tree: ast.Module, source: str):
        self.tree = tree
        self.source = source
        self.lineage: dict[str, frozenset] = {}
        self.objclass: dict[str, str] = {}
        self.strings: dict[str, str] = {}
        self.targets: set[str] = set()
        self.splits: list[_Split] = []
        self.fits: list[_Fit] = []
        self.transforms: list[tuple[int, str, frozenset]] = []   # line, obj, lineage
        self.cv_calls: list[tuple[int, str, frozenset]] = []      # line, X text, lineage
        self.target_aggs: list[tuple[int, str, frozenset]] = []   # line, text, lineage
        self.manual_stats: list[tuple[int, str, frozenset, frozenset]] = []
        self.unshuffled_splits: list[int] = []
        self.early_stop_on_test: list[tuple[int, str]] = []
        self._find_targets()

    # ---------- pre-pass: which column is the target?
    def _str_value(self, node) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return self.strings.get(node.id)
        if isinstance(node, (ast.List, ast.Tuple)) and len(node.elts) == 1:
            return self._str_value(node.elts[0])
        return None

    def _find_targets(self):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                t = node.targets[0]
                if isinstance(t, ast.Name) and isinstance(node.value, ast.Constant) \
                        and isinstance(node.value.value, str):
                    self.strings[t.id] = node.value.value
        for node in ast.walk(self.tree):
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
                continue
            names = _target_names(node.targets[0])
            v = node.value
            yish = any(n.lower() in {"y", "target", "label", "labels", "y_all", "y_full"}
                       for n in names)
            xish = any(n.lower().startswith(("x", "feature")) for n in names)
            if yish and isinstance(v, ast.Subscript):
                s = self._str_value(v.slice)
                if s:
                    self.targets.add(s)
            elif yish and isinstance(v, ast.Attribute) and isinstance(v.value, ast.Name):
                self.targets.add(v.attr)
            elif isinstance(v, ast.Call) and _callee(v) == "pop" and v.args and yish:
                s = self._str_value(v.args[0])
                if s:
                    self.targets.add(s)
            elif xish and isinstance(v, ast.Call) and _callee(v) == "drop":
                arg = _arg(v, 0, "columns", "labels")
                s = self._str_value(arg) if arg is not None else None
                if s:
                    self.targets.add(s)

    # ---------- lineage helpers
    def _lin(self, node: ast.AST | None) -> frozenset:
        if node is None:
            return frozenset()
        out: set = set()
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and n.id in self.lineage:
                out |= self.lineage[n.id]
        return frozenset(out)

    def _obj_class(self, receiver: ast.AST) -> tuple[str | None, str | None]:
        if isinstance(receiver, ast.Name):
            return receiver.id, self.objclass.get(receiver.id)
        if isinstance(receiver, ast.Call):
            return None, _callee(receiver)
        return None, None

    @staticmethod
    def _kind(cls: str | None) -> str:
        if cls in TRANSFORMERS:
            return TRANSFORMERS[cls][0]
        if cls in PIPELINES:
            return "pipeline"
        if cls in SEARCHES:
            return "search"
        return "model"

    # ---------- walk
    def run(self):
        self._body(self.tree.body)

    def _body(self, body):
        for stmt in body:
            self._stmt(stmt)

    def _stmt(self, stmt: ast.stmt):
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            self._body(stmt.body)
            return
        if isinstance(stmt, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With,
                             ast.AsyncWith, ast.Try)):
            if isinstance(stmt, (ast.For, ast.AsyncFor)):
                self._scan_calls(stmt.iter, stmt)
                for n in _target_names(stmt.target):
                    self.lineage[n] = self._lin(stmt.iter)
            for part in ("body", "orelse", "finalbody"):
                self._body(getattr(stmt, part, []) or [])
            for h in getattr(stmt, "handlers", []) or []:
                self._body(h.body)
            return
        self._scan_calls(stmt, stmt)
        self._assign(stmt)

    # ---------- events
    def _scan_calls(self, node: ast.AST, stmt: ast.stmt):
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                self._call(n, stmt)
            elif isinstance(n, ast.Subscript) and isinstance(n.value, ast.Call) \
                    and _callee(n.value) == "groupby":
                col = self._str_value(n.slice)
                if col and col in self.targets:
                    recv = n.value.func.value if isinstance(n.value.func, ast.Attribute) else None
                    self.target_aggs.append((n.lineno, _src(n, 60), self._lin(recv)))

    def _call(self, call: ast.Call, stmt: ast.stmt):
        name = _callee(call)
        if name == "train_test_split":
            ins = frozenset().union(*[self._lin(a) for a in call.args]) if call.args else frozenset()
            self.splits.append(_Split(call.lineno, self._strip_tags(ins)))
            shuffle = _arg(call, 99, "shuffle")
            if not (isinstance(shuffle, ast.Constant) and shuffle.value is False):
                self.unshuffled_splits.append(call.lineno)
            return
        if name in CV_FUNCS:
            X = _arg(call, 1, "X")
            if X is not None:
                self.cv_calls.append((call.lineno, _src(X), self._lin(X)))
            return
        if (name in STAT_METHODS or name in STAT_FUNCS) and isinstance(call.func, ast.Attribute):
            # df['a'].mean()  or  np.mean(df['a'])
            recv_lin = self._lin(call.func.value) | frozenset().union(
                *[self._lin(a) for a in call.args]) if call.args else self._lin(call.func.value)
            if recv_lin and self._is_data_assignment(stmt):
                self.manual_stats.append((call.lineno, _src(call, 50), recv_lin,
                                          self._write_target_lineage(stmt)))
            return
        if not isinstance(call.func, ast.Attribute):
            return
        method = call.func.attr
        obj, cls = self._obj_class(call.func.value)
        if method in FIT_METHODS:
            if cls in IGNORED_TRANSFORMERS:
                return
            X = _arg(call, 0, "X", "raw_documents")
            y = _arg(call, 1, "y")
            lin = self._lin(X) | self._lin(y)
            kind = self._kind(cls)
            self.fits.append(_Fit(call.lineno, obj, cls, kind, method,
                                  _src(X) if X is not None else "", lin))
            es = _arg(call, 99, "eval_set", "validation_data")
            if es is not None and any(t.startswith("TEST#") for t in self._lin(es)):
                self.early_stop_on_test.append((call.lineno, _src(es)))
            if kind == "search":
                self.cv_calls.append((call.lineno, _src(X) if X is not None else "", self._lin(X)))
        elif method == "transform" and obj:
            self.transforms.append((call.lineno, obj, self._lin(_arg(call, 0, "X"))))

    def _write_target_lineage(self, stmt: ast.stmt) -> frozenset:
        if isinstance(stmt, (ast.Assign, ast.AugAssign)):
            tgt = stmt.targets[0] if isinstance(stmt, ast.Assign) else stmt.target
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) \
                and isinstance(stmt.value.func, ast.Attribute):
            tgt = stmt.value.func.value
        else:
            return frozenset()
        base = _base_name(tgt)
        return self.lineage.get(base, frozenset()) if base else frozenset()

    def _is_data_assignment(self, stmt: ast.stmt) -> bool:
        """Is this statement writing derived values back into a tracked dataset?"""
        if isinstance(stmt, ast.Assign):
            tgt = stmt.targets[0]
            val = stmt.value
        elif isinstance(stmt, ast.AugAssign):
            tgt, val = stmt.target, stmt.value
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) \
                and _callee(stmt.value) == "fillna":
            return True
        else:
            return False
        base = _base_name(tgt)
        if base is None or base not in self.lineage:
            return False
        arith = any(isinstance(n, ast.BinOp) for n in ast.walk(val))
        fill = any(isinstance(n, ast.Call) and _callee(n) in {"fillna", "clip", "where"}
                   for n in ast.walk(val))
        return arith or fill

    @staticmethod
    def _strip_tags(lin: frozenset) -> frozenset:
        return frozenset(t for t in lin
                         if not t.startswith(("TEST#", "TRAIN#", "MTRAIN#", "VAL#", "PRE:")))

    # ---------- assignments / lineage propagation
    def _rhs_tags(self, value: ast.AST) -> set:
        tags = set()
        for n in ast.walk(value):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                    and n.func.attr in {"fit_transform", "transform", "fit_resample"}:
                _, cls = self._obj_class(n.func.value)
                if n.func.attr == "fit_resample":
                    kind = "resample"
                elif cls in TRANSFORMERS or cls in PIPELINES:
                    kind = self._kind(cls)
                else:
                    continue  # unknown receiver, e.g. pandas groupby().transform
                tags.add(f"PRE:{kind}#{n.lineno}")
        return tags

    def _assign(self, stmt: ast.stmt):
        if isinstance(stmt, ast.Assign):
            targets, value = stmt.targets, stmt.value
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            targets, value = [stmt.target], stmt.value
        elif isinstance(stmt, ast.AugAssign):
            targets, value = [stmt.target], stmt.value
        else:
            return
        line = stmt.lineno

        # train_test_split: outputs alternate train, test for each input array.
        if isinstance(value, ast.Call) and _callee(value) == "train_test_split":
            parent = self._lin(value)
            # Splitting data that is already a training split yields a
            # validation set, not a new test set.
            held_out = "VAL" if any(t.startswith("TRAIN#") for t in parent) else "TEST"
            for t in targets:
                names = _target_names(t)
                for i, n in enumerate(names):
                    tag = "TRAIN" if i % 2 == 0 else held_out
                    self.lineage[n] = frozenset(parent | {f"{tag}#{line}"})
            return

        # Element-wise tuple assignment: a, b = X[:n], X[n:]
        if len(targets) == 1 and isinstance(targets[0], ast.Tuple) \
                and isinstance(value, ast.Tuple) and len(targets[0].elts) == len(value.elts):
            for t, v in zip(targets[0].elts, value.elts):
                self._assign_one(t, v, line)
            return
        for t in targets:
            self._assign_one(t, value, line)

    def _assign_one(self, t: ast.AST, value: ast.AST, line: int):
        lin = set(self._lin(value)) | self._rhs_tags(value)
        names = _target_names(t)
        if isinstance(value, ast.Call):
            cls = _callee(value)
            for n in names:
                if cls in TRANSFORMERS or cls in IGNORED_TRANSFORMERS or cls in PIPELINES \
                        or cls in SEARCHES or (cls and cls[:1].isupper()
                                               and cls not in _DATA_CTORS):
                    self.objclass[n] = cls
                    self.lineage.pop(n, None)
                    return
        if not names:  # df['col'] = ...  -> df now also depends on the RHS
            base = _base_name(t)
            if base:
                self.lineage[base] = frozenset(self.lineage.get(base, frozenset()) | lin)
            return
        for n in names:
            self._maybe_manual_split(n, value, line, lin)
            # test = pd.read_csv("test.csv"): a held-out set loaded from disk.
            if isinstance(value, ast.Call) and (_callee(value) or "").startswith(("read_", "load")) \
                    and _TEST_NAME.search(n) and not lin:
                lin = {f"{n}@{line}", f"TEST#{line}"}
            if not lin:
                lin = {f"{n}@{line}"}
            self.lineage[n] = frozenset(lin)

    def _maybe_manual_split(self, name: str, value: ast.AST, line: int, lin: set):
        """Recognise X_test = X[n:] / test = df.sample(...) style manual splits."""
        is_slice = isinstance(value, ast.Subscript) or (
            isinstance(value, ast.Call)
            and _callee(value) in {"sample", "drop", "tail", "head", "query", "iloc", "loc"})
        if not is_slice or not lin or any(t.startswith(("TEST#", "TRAIN#", "MTRAIN#"))
                                          for t in lin):
            return
        if _TEST_NAME.search(name):
            self.splits.append(_Split(line, self._strip_tags(frozenset(lin)), manual=True))
            lin.add(f"TEST#{line}")
        elif _TRAIN_NAME.search(name):
            # Provisional: only treated as a training split if a matching
            # manually split test set exists (checked when rules run).
            lin.add(f"MTRAIN#{line}")

    # ---------- rules
    def findings(self) -> list[tuple[str, int, str, str]]:
        """Return (rule, line, severity, message) tuples."""
        out = []

        def overlapping_split(lin: frozenset):
            core = self._strip_tags(lin)
            return [s for s in self.splits if core & s.inputs]

        def has(lin, prefix):
            return any(t.startswith(prefix) for t in lin)

        def is_train(lin):
            if has(lin, "TRAIN#"):
                return True
            core = self._strip_tags(lin)
            return has(lin, "MTRAIN#") and any(s.manual and core & s.inputs for s in self.splits)

        for f in self.fits:
            label = f.cls or (f.obj or "estimator")
            # LS103: fitting on anything that contains test rows
            if has(f.lineage, "TEST#"):
                out.append(("LS103", f.line, "high",
                            f"`{label}.{f.method}({f.arg_text})` is fitted on data that "
                            "contains the test set. (If this is an unlabelled competition "
                            "test set, this is transductive preprocessing: check that your "
                            "validation score was computed without it.)"
                            if f.kind not in ("model", "search", "pipeline") else
                            f"`{label}.{f.method}({f.arg_text})` is fitted on data that "
                            "contains the test set."))
                continue
            if f.kind in ("model", "search", "pipeline", "unknown") or is_train(f.lineage):
                continue
            later = [s for s in overlapping_split(f.lineage) if s.line > f.line]
            sev = TRANSFORMERS.get(f.cls, ("", "medium"))[1]
            if later:
                rule = "LS102" if f.kind == "resample" else "LS101"
                out.append((rule, f.line, sev,
                            f"`{label}.{f.method}({f.arg_text})` learns from the full "
                            f"dataset before the train/test split on line {later[0].line}."))
                continue
            earlier = [s for s in overlapping_split(f.lineage) if s.line < f.line]
            if earlier and f.obj and any(o == f.obj and has(lin, "TEST#") and ln > f.line
                                         for ln, o, lin in self.transforms):
                out.append(("LS101", f.line, sev,
                            f"`{label}` is fitted on the full dataset `{f.arg_text}` and then "
                            "applied to the test split, so it has seen the test rows."))

        for line, text, lin, tgt_lin in self.manual_stats:
            if has(lin, "TEST#"):
                out.append(("LS103", line, "medium",
                            f"`{text}` is computed on data that includes the test set and "
                            "used to transform it; compute it on the training set instead."))
                continue
            if is_train(lin):
                continue
            if (has(tgt_lin, "TRAIN#") or has(tgt_lin, "TEST#")) and \
                    [s for s in overlapping_split(lin) if s.line < line]:
                out.append(("LS101", line, "medium",
                            f"`{text}` is computed on the full dataset (including test rows) "
                            "and written into a train/test split."))
                continue
            later = [s for s in overlapping_split(lin) if s.line > line]
            if later:
                out.append(("LS101", line, "medium",
                            f"`{text}` is computed on the full dataset before the split on "
                            f"line {later[0].line} and used to transform the data."))

        for line, text, lin in self.cv_calls:
            kinds = {t.split(":")[1].split("#")[0] for t in lin if t.startswith("PRE:")}
            if kinds:
                sev = "high" if kinds & HIGH_RISK_KINDS else "low"
                out.append(("LS104", line, sev,
                            f"Cross-validation runs on `{text}`, which was already "
                            f"transformed ({', '.join(sorted(kinds))}) outside the CV loop."))

        for line, text, lin in self.target_aggs:
            later = [s for s in overlapping_split(lin) if s.line > line] if lin else \
                [s for s in self.splits if s.line > line]
            if later:
                out.append(("LS105", line, "high",
                            f"`{text}` aggregates the target over all rows before the split "
                            f"on line {later[0].line}."))

        for line, text in self.early_stop_on_test:
            out.append(("LS103", line, "medium",
                        f"Early stopping / validation uses `{text}`, which contains the test "
                        "set, so the test set influences training. Use a separate "
                        "validation split."))

        if _TIME_HINT.search(self.source):
            for line in self.unshuffled_splits:
                out.append(("LS106", line, "low",
                            "The data appears to contain dates, but this split shuffles rows."))
        return out


# --------------------------------------------------------------------------- public API
def _suppressions(source: str) -> dict[int, set[str] | None]:
    sup: dict[int, set[str] | None] = {}
    for i, ln in enumerate(source.splitlines(), start=1):
        m = _SUPPRESS.search(ln)
        if m:
            sup[i] = {r.strip().upper() for r in m.group(1).split(",")} if m.group(1) else None
    return sup


def scan_source(source: str, path: str = "<string>",
                line_map: list[tuple[int | None, int]] | None = None) -> ScanResult:
    """Scan Python source text and return findings."""
    res = ScanResult(path=path)
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        res.warnings.append(f"could not parse: {e.msg} (line {e.lineno})")
        return res
    an = _Analyzer(tree, source)
    an.run()
    sup = _suppressions(source)
    seen = set()
    for rule_id, line, sev, msg in an.findings():
        if (rule_id, line) in seen:
            continue
        seen.add((rule_id, line))
        if line in sup and (sup[line] is None or rule_id in sup[line]):
            continue
        r = RULES[rule_id]
        cell, local = (line_map[line - 1] if line_map and line - 1 < len(line_map)
                       else (None, line))
        res.findings.append(Finding(rule_id, r.name, sev, path, local, msg, r.fix,
                                    r.taxonomy, cell))
    res.findings.sort(key=lambda f: ((f.cell or 0), f.line, f.rule))
    return res


def scan_file(path: str | Path) -> ScanResult:
    p = Path(path)
    source, lmap, warnings = _load_source(p)
    res = scan_source(source, str(p), lmap)
    res.warnings = warnings + res.warnings
    return res


def iter_paths(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(q for q in p.rglob("*") if q.suffix in {".py", ".ipynb"}
                          and ".ipynb_checkpoints" not in q.parts)
        elif p.suffix in {".py", ".ipynb"}:
            out.append(p)
    return out


def at_least(findings: list[Finding], severity: str) -> list[Finding]:
    return [f for f in findings if SEVERITY_ORDER[f.severity] >= SEVERITY_ORDER[severity]]
