"""A minimal table model rendered as Markdown, CSV or JSON."""

from __future__ import annotations

import csv
import io
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

Cell = str | int | float


@dataclass(frozen=True)
class Table:
    """Rows of cells under named columns; ``align`` holds ``"l"``/``"r"`` per column."""

    columns: list[str]
    rows: list[list[Cell]] = field(default_factory=list)
    align: list[str] | None = None
    float_format: str = "{:,.2f}"
    csv_columns: list[str] | None = None  # machine-friendly header for CSV/JSON

    def __post_init__(self) -> None:
        for row in self.rows:
            if len(row) != len(self.columns):
                raise ValueError(f"row {row!r} does not match columns {self.columns!r}")
        if self.csv_columns is not None and len(self.csv_columns) != len(self.columns):
            raise ValueError("csv_columns must match columns")

    @property
    def machine_columns(self) -> list[str]:
        """Column names used for CSV and JSON."""
        return self.csv_columns or self.columns

    def _fmt(self, value: Cell) -> str:
        if isinstance(value, float):
            if math.isnan(value):
                return "-"
            return self.float_format.format(value)
        return str(value)

    def to_markdown(self) -> str:
        """GitHub-flavoured Markdown table."""
        align = self.align or [
            "r" if self.rows and isinstance(self.rows[0][i], int | float) else "l"
            for i in range(len(self.columns))
        ]
        sep = ["---:" if a == "r" else ":---" for a in align]
        lines = [
            "| " + " | ".join(self.columns) + " |",
            "| " + " | ".join(sep) + " |",
        ]
        lines += ["| " + " | ".join(self._fmt(v) for v in row) + " |" for row in self.rows]
        return "\n".join(lines)

    def to_csv(self) -> str:
        """RFC 4180 CSV with full-precision numbers (empty cell for NaN)."""
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(self.machine_columns)
        for row in self.rows:
            writer.writerow(
                ["" if isinstance(v, float) and math.isnan(v) else _csv_value(v) for v in row]
            )
        return buffer.getvalue()

    def to_records(self) -> list[dict[str, Cell | None]]:
        """List of dicts (NaN -> None) for JSON output."""
        return [
            {
                col: (None if isinstance(v, float) and math.isnan(v) else v)
                for col, v in zip(self.machine_columns, row, strict=True)
            }
            for row in self.rows
        ]

    def to_json(self) -> str:
        """JSON array of records."""
        return json.dumps(self.to_records(), indent=2)

    def render(self, fmt: str) -> str:
        """Render as ``markdown``/``table``, ``csv`` or ``json``."""
        if fmt in ("markdown", "table"):
            return self.to_markdown()
        if fmt == "csv":
            return self.to_csv()
        if fmt == "json":
            return self.to_json()
        raise ValueError(f"unknown format {fmt!r}")

    def write_csv(self, path: Path) -> Path:
        """Write the CSV rendering to ``path``."""
        path.write_text(self.to_csv(), encoding="utf-8")
        return path


def _csv_value(value: Cell) -> str:
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".") if value != 0 else "0"
    return str(value)


def bullet_list(items: Sequence[str]) -> str:
    """Markdown bullet list (``- none`` when empty)."""
    return "\n".join(f"- {item}" for item in items) if items else "- none"
