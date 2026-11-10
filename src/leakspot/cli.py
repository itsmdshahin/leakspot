"""Command-line interface.

    leakspot scan PATH...              static scan of .py / .ipynb files or folders
    leakspot audit TRAIN TEST -t COL   data-level audit of a train/test pair
    leakspot rules                     list the rules

Exit codes: 0 = nothing at or above --fail-on, 1 = leakage found, 2 = usage error.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from .rules import RULES, SEVERITY_ORDER
from .scanner import at_least, iter_paths, scan_file

SEVERITIES = ["low", "medium", "high", "never"]


def _read_table(path: str):
    import pandas as pd

    p = Path(path)
    if p.suffix in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    if p.suffix in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    return pd.read_csv(p)


@click.group()
@click.version_option(package_name="leakspot")
def main():
    """LeakSpot: find data leakage in ML code and data."""


@main.command()
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--format", "fmt", type=click.Choice(["text", "json", "github"]),
              default="text", show_default=True,
              help="'github' prints GitHub Actions annotations.")
@click.option("--fail-on", type=click.Choice(SEVERITIES), default="medium", show_default=True,
              help="Exit with code 1 if a finding of this severity or higher is found.")
@click.option("--select", default="", help="Only these rules, e.g. LS101,LS103.")
@click.option("--ignore", default="", help="Skip these rules.")
def scan(paths, fmt, fail_on, select, ignore):
    """Statically scan Python files and notebooks for leakage."""
    sel = {r.strip().upper() for r in select.split(",") if r.strip()}
    ign = {r.strip().upper() for r in ignore.split(",") if r.strip()}
    files = iter_paths(list(paths))
    findings, warnings = [], []
    for f in files:
        res = scan_file(f)
        warnings += [f"{f}: {w}" for w in res.warnings]
        findings += [x for x in res.findings
                     if (not sel or x.rule in sel) and x.rule not in ign]

    if fmt == "json":
        click.echo(json.dumps({"files_scanned": len(files),
                               "findings": [x.to_dict() for x in findings],
                               "warnings": warnings}, indent=2))
    elif fmt == "github":
        level = {"high": "error", "medium": "warning", "low": "notice"}
        for x in findings:
            click.echo(f"::{level[x.severity]} file={x.path},line={x.line},"
                       f"title={x.rule} {x.name}::{x.message} Fix: {x.fix}")
    else:
        for x in findings:
            click.echo(f"{x.location()}: {x.rule} [{x.severity}] {x.name}")
            click.echo(f"    {x.message}")
            click.echo(f"    fix: {x.fix}")
        for w in warnings:
            click.echo(f"warning: {w}", err=True)
        n = {s: sum(1 for x in findings if x.severity == s) for s in ("high", "medium", "low")}
        click.echo(f"\n{len(files)} file(s) scanned: {n['high']} high, {n['medium']} medium, "
                   f"{n['low']} low.")
    if fail_on != "never" and at_least(findings, fail_on):
        sys.exit(1)


@main.command()
@click.argument("train", type=click.Path(exists=True))
@click.argument("test", type=click.Path(exists=True))
@click.option("-t", "--target", required=True, help="Target / label column.")
@click.option("--time-col", default=None, help="Timestamp column (time-based split).")
@click.option("--group-col", default=None, help="Entity column that must not cross the split.")
@click.option("--id-cols", default=None,
              help="Comma-separated identifier columns. Default: auto-detect.")
@click.option("--format", "fmt", type=click.Choice(["text", "json"]), default="text",
              show_default=True)
@click.option("--fail-on", type=click.Choice(SEVERITIES), default="high", show_default=True)
def audit(train, test, target, time_col, group_col, id_cols, fmt, fail_on):
    """Audit a train/test pair of CSV/Parquet files for data-level leakage."""
    from .audit import audit as run_audit

    ids = [c.strip() for c in id_cols.split(",")] if id_cols else None
    rep = run_audit(_read_table(train), _read_table(test), target,
                    time_col=time_col, group_col=group_col, id_cols=ids)
    click.echo(json.dumps(rep.to_dict(), indent=2) if fmt == "json" else rep.to_text())
    if fail_on != "never" and any(SEVERITY_ORDER[f.severity] >= SEVERITY_ORDER[fail_on]
                                  for f in rep.findings):
        sys.exit(1)


@main.command()
def rules():
    """List the static-analysis rules."""
    for r in RULES.values():
        click.echo(f"{r.id}  {r.name}\n      {r.summary}\n      fix: {r.fix}\n"
                   f"      taxonomy: {r.taxonomy}\n")


if __name__ == "__main__":
    main()
