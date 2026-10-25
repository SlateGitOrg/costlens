"""Command-line interface, end to end."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from costlens.cli import main


def test_generate_then_every_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    export = tmp_path / "export"
    assert (
        main(
            [
                "generate",
                "--out",
                str(export),
                "--weeks",
                "2",
                "--step",
                "1h",
                "--seed",
                "3",
                "--chunk-days",
                "7",
            ]
        )
        == 0
    )
    assert "export files" in capsys.readouterr().out

    assert main(["validate", str(export)]) == 0
    out = capsys.readouterr().out
    assert "cluster       demo-shop" in out and "warnings      0" in out

    assert (
        main(
            [
                "allocate",
                str(export),
                "--group-by",
                "team",
                "--idle",
                "keep",
                "--basis",
                "request",
                "--shared-namespaces",
                "kube-system,monitoring",
                "--shared-strategy",
                "even",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "| __idle__ |" in out and "Idle capacity" in out

    out_csv = tmp_path / "alloc.csv"
    assert main(["allocate", str(export), "--format", "csv", "--out", str(out_csv)]) == 0
    rows = list(csv.DictReader(out_csv.open()))
    assert rows[0].keys() >= {"tenant", "total_usd"}

    assert main(["allocate", str(export), "--format", "json"]) == 0
    assert isinstance(json.loads(capsys.readouterr().out), list)

    config = tmp_path / "alloc.toml"
    config.write_text('[allocation]\ngroup_by = "label:app"\nidle = "per_node"\n')
    assert main(["allocate", str(export), "--config", str(config), "--format", "csv"]) == 0
    assert "payments-api" in capsys.readouterr().out

    assert main(["recommend", str(export)]) == 0
    assert "Total request-cost change" in capsys.readouterr().out
    assert (
        main(
            [
                "recommend",
                str(export),
                "--format",
                "csv",
                "--no-oom-bump",
                "--cpu-percentile",
                "0.95",
                "--margin",
                "0.2",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.startswith("workload,container,")

    assert (
        main(
            [
                "backtest",
                str(export),
                "--train-weeks",
                "1",
                "--validation-weeks",
                "0.5",
                "--test-weeks",
                "0.5",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "selected on validation" in out and "uncensored demand" in out
    assert (
        main(
            [
                "backtest",
                str(export),
                "--train-weeks",
                "1",
                "--validation-weeks",
                "0.5",
                "--test-weeks",
                "0.5",
                "--format",
                "json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)[0]["policy"] == "current requests"

    report = tmp_path / "report"
    assert main(["report", str(export), "--out", str(report), "--group-by-list", "team"]) == 0
    capsys.readouterr()
    # default backtest needs 6 weeks; the report still renders without it
    assert (report / "showback.md").exists()
    assert not (report / "backtest_test.csv").exists()
    assert main(["report", str(export), "--out", str(tmp_path / "r2"), "--no-backtest"]) == 0
    assert (tmp_path / "r2" / "allocation_namespace.csv").exists()


def test_queries(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["queries", "--step", "15m"]) == 0
    out = capsys.readouterr().out
    assert "container_cpu_usage_seconds_total" in out and "[900s]" in out
    assert main(["queries", "--script", "--step", "5m", "--chunk", "60d"]) == 0
    assert "CHUNK=3300000" in capsys.readouterr().out


def test_errors_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate", str(tmp_path)]) == 2
    assert "no manifest" in capsys.readouterr().err
    assert main(["generate", "--out", str(tmp_path / "x"), "--step", "7m"]) == 2
    with pytest.raises(SystemExit):
        main(["allocate"])
