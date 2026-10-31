"""Model Context Protocol server exposing DataGuard's validation engine as
tools any MCP client (Claude Desktop, another agent, etc.) can call.

This is the piece most early-career resumes won't have: hands-on MCP server
authorship, not just "used an MCP client." Three tools are exposed:

    validate_dataset(file_path, key_columns)  -> full validation report (JSON)
    get_critical_findings(file_path)          -> just the critical findings
    explain_finding(file_path, rule, column)   -> plain-English explanation

Run it:
    python -m dataguard.mcp_server

Then point an MCP client at it (stdio transport). For Claude Desktop, add
to claude_desktop_config.json:
    {
      "mcpServers": {
        "dataguard": {
          "command": "python",
          "args": ["-m", "dataguard.mcp_server"],
          "cwd": "/absolute/path/to/dataguard-ai"
        }
      }
    }
"""

from __future__ import annotations

import json

import pandas as pd
from mcp.server.mcpserver import MCPServer

from dataguard.validator import DataValidator

server = MCPServer(
    name="dataguard",
    description="Data quality validation and anomaly detection over CSV datasets.",
)


def _load_and_validate(file_path: str, key_columns: list[str] | None = None):
    df = pd.read_csv(file_path)
    validator = DataValidator(dataset_name=file_path)
    return validator.validate(df, key_columns=key_columns)


@server.tool()
def validate_dataset(file_path: str, key_columns: list[str] | None = None) -> str:
    """Run the full DataGuard validation suite against a CSV file and
    return a JSON report: row/column counts, critical/warning counts, and
    every individual finding with its severity, affected column, row
    count, and up to 5 sample rows.

    Args:
        file_path: Path to a CSV file, readable from where this server runs.
        key_columns: Optional list of column names expected to be unique
            together (e.g. ["order_id"]). Enables duplicate-key checking.
    """
    report = _load_and_validate(file_path, key_columns)
    return json.dumps(report.to_dict(), default=str, indent=2)


@server.tool()
def get_critical_findings(file_path: str, key_columns: list[str] | None = None) -> str:
    """Run validation and return only the critical-severity findings, as
    JSON. Use this when you just need to know what would block a dataset
    from being trusted downstream, without the full report.

    Args:
        file_path: Path to a CSV file.
        key_columns: Optional unique-key column(s) for duplicate-key checks.
    """
    report = _load_and_validate(file_path, key_columns)
    critical = [f.to_dict() for f in report.findings if f.severity == "critical"]
    return json.dumps({"dataset": file_path, "critical_findings": critical}, default=str, indent=2)


@server.tool()
def explain_finding(file_path: str, rule: str, column: str | None = None) -> str:
    """Run validation and return a plain-English explanation of a specific
    finding (matched by rule name and, optionally, column), including what
    it means and a suggested next step. Useful for a non-technical
    stakeholder asking "what does 'type_mismatch on order_date' mean?"

    Args:
        file_path: Path to a CSV file.
        rule: The rule name to explain (e.g. "missing_values", "duplicate_keys",
            "type_mismatch", "out_of_range", "statistical_outlier", "duplicate_rows").
        column: Optional column name to narrow the match.
    """
    report = _load_and_validate(file_path)
    matches = [
        f for f in report.findings if f.rule == rule and (column is None or f.column == column)
    ]
    if not matches:
        return json.dumps({"found": False, "message": f"No finding matching rule='{rule}', column='{column}'."})

    explanations = {
        "missing_values": "Some rows are missing a value in this column. Downstream "
        "calculations or joins on this field may silently drop or misrepresent these rows.",
        "duplicate_rows": "These rows are identical across every column. This usually "
        "means the same record was ingested more than once (e.g. a re-run pipeline).",
        "duplicate_keys": "Multiple rows share a value in a column (or columns) that's "
        "expected to be unique. This can cause incorrect joins, double-counting, or "
        "overwritten records.",
        "type_mismatch": "Some values in this column don't match the expected data type "
        "(e.g. text in a date column). These rows will likely fail or be silently "
        "coerced in downstream numeric/date processing.",
        "out_of_range": "Some values fall outside the range considered valid for this "
        "field. This often indicates a data-entry error or a unit mismatch (e.g. cents "
        "vs. dollars).",
        "statistical_outlier": "These values are statistical outliers (far from the "
        "column's typical range). Not necessarily wrong, but worth a manual look.",
    }

    out = []
    for f in matches:
        out.append(
            {
                "rule": f.rule,
                "column": f.column,
                "severity": f.severity,
                "affected_rows": f.row_count,
                "detail": f.message,
                "what_it_means": explanations.get(rule, "No canned explanation for this rule."),
                "suggested_next_step": "Flag for human review before downstream use."
                if f.severity == "critical"
                else "Monitor; address if the rate increases.",
            }
        )
    return json.dumps({"found": True, "explanations": out}, default=str, indent=2)


if __name__ == "__main__":
    server.run()
