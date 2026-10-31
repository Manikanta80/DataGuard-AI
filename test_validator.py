"""Unit tests for the validation engine — run with: pytest tests/"""

import pandas as pd
import pytest

from dataguard.validator import DataValidator


def test_missing_values_detected():
    df = pd.DataFrame({"a": [1, 2, None], "b": [1, 2, 3]})
    report = DataValidator("t").validate(df)
    findings = [f for f in report.findings if f.rule == "missing_values"]
    assert len(findings) == 1
    assert findings[0].column == "a"
    assert findings[0].row_count == 1


def test_duplicate_rows_detected():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    report = DataValidator("t").validate(df)
    findings = [f for f in report.findings if f.rule == "duplicate_rows"]
    assert len(findings) == 1
    assert findings[0].row_count == 2


def test_duplicate_keys_detected():
    df = pd.DataFrame({"id": [1, 1, 2], "val": ["x", "y", "z"]})
    report = DataValidator("t").validate(df, key_columns=["id"])
    findings = [f for f in report.findings if f.rule == "duplicate_keys"]
    assert len(findings) == 1
    assert findings[0].row_count == 2
    assert findings[0].severity == "critical"


def test_duplicate_keys_missing_column():
    df = pd.DataFrame({"a": [1, 2]})
    report = DataValidator("t").validate(df, key_columns=["id"])
    findings = [f for f in report.findings if f.rule == "duplicate_keys"]
    assert len(findings) == 1
    assert "not found" in findings[0].message


def test_type_mismatch_numeric():
    df = pd.DataFrame({"price": ["10", "20", "oops"]})
    report = DataValidator("t").validate(df, expected_types={"price": "numeric"})
    findings = [f for f in report.findings if f.rule == "type_mismatch"]
    assert len(findings) == 1
    assert findings[0].row_count == 1


def test_type_mismatch_datetime():
    df = pd.DataFrame({"d": ["2026-01-01", "not-a-date"]})
    report = DataValidator("t").validate(df, expected_types={"d": "datetime"})
    findings = [f for f in report.findings if f.rule == "type_mismatch"]
    assert len(findings) == 1


def test_out_of_range():
    df = pd.DataFrame({"qty": [1, 5, 9999]})
    report = DataValidator("t").validate(df, ranges={"qty": (0, 100)})
    findings = [f for f in report.findings if f.rule == "out_of_range"]
    assert len(findings) == 1
    assert findings[0].row_count == 1


def test_outlier_zscore():
    # Needs a reasonably sized sample: a single extreme value among very few
    # points inflates the std enough to mask its own z-score (a known
    # limitation of z-score outlier detection — see validator.py docstring).
    normal = [10, 11, 9, 10, 12, 11, 9, 10, 11, 10] * 3
    df = pd.DataFrame({"x": normal + [500]})
    report = DataValidator("t").validate(df, outlier_columns=["x"])
    findings = [f for f in report.findings if f.rule == "statistical_outlier"]
    assert len(findings) == 1
    assert findings[0].severity == "info"


def test_outlier_zscore_single_extreme_value_in_tiny_sample_is_masked():
    """Documents the z-score masking effect rather than hiding it: with
    very few points, one huge outlier inflates the std so much that its own
    z-score can fall below the threshold. This is why `statistical_outlier`
    is marked severity='info' (a hint, not a hard rule) — flag it in code
    review, don't silently trust it on small samples."""
    df = pd.DataFrame({"x": [10, 11, 9, 10, 12, 500]})
    report = DataValidator("t").validate(df, outlier_columns=["x"])
    findings = [f for f in report.findings if f.rule == "statistical_outlier"]
    assert len(findings) == 0  # masked — not a bug, a documented limitation


def test_clean_dataset_has_no_critical_findings():
    df = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
    report = DataValidator("t").validate(df, key_columns=["id"])
    assert report.critical_count == 0
    assert report.accuracy_estimate() == 100.0


def test_custom_rule_is_applied():
    df = pd.DataFrame({"status": ["Delivered", "Unknown", "Pending"]})

    def rule_valid_status(d):
        from dataguard.validator import Finding

        valid = {"Delivered", "Pending", "Cancelled"}
        bad = d[~d["status"].isin(valid)]
        if bad.empty:
            return []
        return [
            Finding(
                rule="invalid_status",
                severity="critical",
                column="status",
                row_count=len(bad),
                message=f"{len(bad)} row(s) have a status outside {valid}.",
            )
        ]

    validator = DataValidator("t")
    validator.add_rule("invalid_status", rule_valid_status)
    report = validator.validate(df)
    findings = [f for f in report.findings if f.rule == "invalid_status"]
    assert len(findings) == 1
    assert findings[0].row_count == 1


def test_custom_rule_error_is_caught_not_raised():
    df = pd.DataFrame({"a": [1, 2]})

    def broken_rule(d):
        raise ValueError("boom")

    validator = DataValidator("t")
    validator.add_rule("broken", broken_rule)
    report = validator.validate(df)  # should not raise
    findings = [f for f in report.findings if f.rule == "broken"]
    assert len(findings) == 1
    assert "boom" in findings[0].message


def test_report_to_dict_is_json_serializable():
    import json

    df = pd.DataFrame({"a": [1, None, 3]})
    report = DataValidator("t").validate(df)
    json.dumps(report.to_dict(), default=str)  # should not raise
