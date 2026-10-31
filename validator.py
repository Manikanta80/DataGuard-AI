"""Rule-based data validation & anomaly detection engine.

This is the one module in the project with zero external-API dependency —
it's pure Pandas, so it runs (and is tested) completely offline. Everything
downstream (the LangChain chain, the LangGraph workflow, the MCP server)
wraps around the structured `Finding` objects this module produces.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd


@dataclass
class Finding:
    """A single data-quality issue, in a structure an LLM can summarize
    and a human can audit."""

    rule: str
    severity: str  # "critical" | "warning" | "info"
    column: str | None
    row_count: int
    message: str
    sample_rows: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationReport:
    dataset_name: str
    row_count: int
    column_count: int
    generated_at: str
    findings: list[Finding]

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "critical")

    @property
    def warning_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "warning")

    def accuracy_estimate(self) -> float:
        """Share of rows with zero critical/warning findings attached to them,
        as a simple proxy metric — mirrors the kind of 'accuracy %' figure
        used in manual QA reporting, but computed rather than hand-tallied."""
        if self.row_count == 0:
            return 100.0
        bad_rows = set()
        for f in self.findings:
            if f.severity in ("critical", "warning"):
                for r in f.sample_rows:
                    if "_row_index" in r:
                        bad_rows.add(r["_row_index"])
        # sample_rows is capped (see MAX_SAMPLE_ROWS), so for the headline
        # number we fall back to the reported row_count per finding when it
        # exceeds the sample, treating each such finding as fully distinct
        # rows (a conservative / worst-case estimate).
        counted = max(len(bad_rows), sum(f.row_count for f in self.findings if f.severity in ("critical", "warning")))
        counted = min(counted, self.row_count)
        return round(100.0 * (self.row_count - counted) / self.row_count, 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "generated_at": self.generated_at,
            "critical_count": self.critical_count,
            "warning_count": self.warning_count,
            "accuracy_estimate_pct": self.accuracy_estimate(),
            "findings": [f.to_dict() for f in self.findings],
        }


MAX_SAMPLE_ROWS = 5


class DataValidator:
    """Runs a configurable set of rule checks against a DataFrame.

    Built-in checks: missing values, duplicate rows, duplicate keys, type
    mismatches, out-of-range numeric values, and z-score outliers. Callers
    can also register custom business-rule functions.
    """

    def __init__(self, dataset_name: str = "dataset"):
        self.dataset_name = dataset_name
        self._custom_rules: list[tuple[str, Callable[[pd.DataFrame], list[Finding]]]] = []

    def add_rule(self, name: str, fn: Callable[[pd.DataFrame], list[Finding]]) -> None:
        """Register a custom business-rule function.

        `fn` receives the full DataFrame and returns a list of Findings.
        """
        self._custom_rules.append((name, fn))

    # ---- built-in checks -------------------------------------------------

    def _check_missing_values(self, df: pd.DataFrame) -> list[Finding]:
        findings = []
        for col in df.columns:
            n_missing = int(df[col].isna().sum())
            if n_missing > 0:
                sample = df[df[col].isna()].head(MAX_SAMPLE_ROWS)
                findings.append(
                    Finding(
                        rule="missing_values",
                        severity="warning" if n_missing < len(df) * 0.1 else "critical",
                        column=col,
                        row_count=n_missing,
                        message=f"Column '{col}' has {n_missing} missing value(s) "
                        f"({n_missing / len(df):.1%} of rows).",
                        sample_rows=self._rows_with_index(sample),
                    )
                )
        return findings

    def _check_duplicate_rows(self, df: pd.DataFrame) -> list[Finding]:
        dup_mask = df.duplicated(keep=False)
        n_dup = int(dup_mask.sum())
        if n_dup == 0:
            return []
        sample = df[dup_mask].head(MAX_SAMPLE_ROWS)
        return [
            Finding(
                rule="duplicate_rows",
                severity="warning",
                column=None,
                row_count=n_dup,
                message=f"{n_dup} fully duplicate row(s) found.",
                sample_rows=self._rows_with_index(sample),
            )
        ]

    def _check_duplicate_keys(self, df: pd.DataFrame, key_columns: list[str]) -> list[Finding]:
        missing_keys = [c for c in key_columns if c not in df.columns]
        if missing_keys:
            return [
                Finding(
                    rule="duplicate_keys",
                    severity="critical",
                    column=", ".join(missing_keys),
                    row_count=0,
                    message=f"Key column(s) not found in dataset: {missing_keys}.",
                )
            ]
        dup_mask = df.duplicated(subset=key_columns, keep=False)
        n_dup = int(dup_mask.sum())
        if n_dup == 0:
            return []
        sample = df[dup_mask].head(MAX_SAMPLE_ROWS)
        return [
            Finding(
                rule="duplicate_keys",
                severity="critical",
                column=", ".join(key_columns),
                row_count=n_dup,
                message=f"{n_dup} row(s) share a duplicate key on {key_columns} "
                f"(expected unique).",
                sample_rows=self._rows_with_index(sample),
            )
        ]

    def _check_type_consistency(self, df: pd.DataFrame, expected_types: dict[str, str]) -> list[Finding]:
        findings = []
        for col, expected in expected_types.items():
            if col not in df.columns:
                continue
            if expected == "numeric":
                coerced = pd.to_numeric(df[col], errors="coerce")
                bad_mask = coerced.isna() & df[col].notna()
            elif expected == "datetime":
                coerced = pd.to_datetime(df[col], errors="coerce")
                bad_mask = coerced.isna() & df[col].notna()
            else:
                continue
            n_bad = int(bad_mask.sum())
            if n_bad > 0:
                sample = df[bad_mask].head(MAX_SAMPLE_ROWS)
                findings.append(
                    Finding(
                        rule="type_mismatch",
                        severity="critical",
                        column=col,
                        row_count=n_bad,
                        message=f"Column '{col}' expected to be {expected}; "
                        f"{n_bad} value(s) don't parse as {expected}.",
                        sample_rows=self._rows_with_index(sample),
                    )
                )
        return findings

    def _check_range(self, df: pd.DataFrame, ranges: dict[str, tuple[float, float]]) -> list[Finding]:
        findings = []
        for col, (lo, hi) in ranges.items():
            if col not in df.columns:
                continue
            numeric = pd.to_numeric(df[col], errors="coerce")
            bad_mask = (numeric < lo) | (numeric > hi)
            bad_mask = bad_mask.fillna(False)
            n_bad = int(bad_mask.sum())
            if n_bad > 0:
                sample = df[bad_mask].head(MAX_SAMPLE_ROWS)
                findings.append(
                    Finding(
                        rule="out_of_range",
                        severity="critical",
                        column=col,
                        row_count=n_bad,
                        message=f"Column '{col}' has {n_bad} value(s) outside expected "
                        f"range [{lo}, {hi}].",
                        sample_rows=self._rows_with_index(sample),
                    )
                )
        return findings

    def _check_outliers_zscore(self, df: pd.DataFrame, columns: list[str] | None = None, threshold: float = 3.0) -> list[Finding]:
        """Flags values more than `threshold` standard deviations from the
        column mean.

        Known limitation: a single extreme value in a small sample inflates
        the std enough to mask its own z-score (the outlier drags the
        "normal" range toward itself). This is exactly why findings from
        this check are marked severity='info' rather than 'critical' or
        'warning' elsewhere in this class — treat it as a hint worth a
        manual look, not a reliable hard rule, especially on datasets under
        ~20-30 rows. A more robust approach for small/skewed samples would
        use a median-based measure (e.g. MAD) instead of mean/std.
        """
        findings = []
        numeric_cols = columns or df.select_dtypes(include="number").columns.tolist()
        for col in numeric_cols:
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            std = series.std()
            if not std or pd.isna(std) or std == 0:
                continue
            z = (series - series.mean()) / std
            bad_mask = z.abs() > threshold
            n_bad = int(bad_mask.sum())
            if n_bad > 0:
                sample = df[bad_mask].head(MAX_SAMPLE_ROWS)
                findings.append(
                    Finding(
                        rule="statistical_outlier",
                        severity="info",
                        column=col,
                        row_count=n_bad,
                        message=f"Column '{col}' has {n_bad} value(s) beyond "
                        f"{threshold} standard deviations from the mean (possible anomalies).",
                        sample_rows=self._rows_with_index(sample),
                    )
                )
        return findings

    @staticmethod
    def _rows_with_index(sample: pd.DataFrame) -> list[dict[str, Any]]:
        out = []
        for idx, row in sample.iterrows():
            d = row.to_dict()
            d = {k: (None if pd.isna(v) else v) for k, v in d.items()}
            d["_row_index"] = int(idx)
            out.append(d)
        return out

    # ---- public API --------------------------------------------------

    def validate(
        self,
        df: pd.DataFrame,
        key_columns: list[str] | None = None,
        expected_types: dict[str, str] | None = None,
        ranges: dict[str, tuple[float, float]] | None = None,
        outlier_columns: list[str] | None = None,
        run_outlier_check: bool = True,
    ) -> ValidationReport:
        findings: list[Finding] = []
        findings += self._check_missing_values(df)
        findings += self._check_duplicate_rows(df)
        if key_columns:
            findings += self._check_duplicate_keys(df, key_columns)
        if expected_types:
            findings += self._check_type_consistency(df, expected_types)
        if ranges:
            findings += self._check_range(df, ranges)
        if run_outlier_check:
            findings += self._check_outliers_zscore(df, outlier_columns)

        for name, fn in self._custom_rules:
            try:
                findings += fn(df)
            except Exception as e:  # a bad custom rule shouldn't crash the run
                findings.append(
                    Finding(
                        rule=name,
                        severity="warning",
                        column=None,
                        row_count=0,
                        message=f"Custom rule '{name}' raised an error: {e}",
                    )
                )

        return ValidationReport(
            dataset_name=self.dataset_name,
            row_count=len(df),
            column_count=len(df.columns),
            generated_at=datetime.now(timezone.utc).isoformat(),
            findings=findings,
        )
