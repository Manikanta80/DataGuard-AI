"""LangGraph orchestration: wires validation, LLM summarization, and a
conditional human-review step into a single graph.

Graph shape:

    ingest -> validate -> summarize -> (conditional) -> report
                                           |
                                           +-- critical findings exist --> human_review -> report
                                           +-- no critical findings ------------------------> report

Run it with `python -m dataguard.workflow data/sample_orders.csv` (see
`__main__` block) or import `build_graph()` / `run_pipeline()` directly.
"""

from __future__ import annotations

import os
import sys
from typing import Any, TypedDict

import pandas as pd
from langgraph.graph import StateGraph, END

from dataguard.validator import DataValidator, ValidationReport


class PipelineState(TypedDict, total=False):
    file_path: str
    dataset_name: str
    key_columns: list[str]
    df: Any  # pd.DataFrame — not serializable, fine for in-process graphs
    report: ValidationReport
    summary: str
    needs_human_review: bool
    human_review_note: str
    final_report_text: str


# ---- nodes -----------------------------------------------------------


def node_ingest(state: PipelineState) -> PipelineState:
    df = pd.read_csv(state["file_path"])
    return {"df": df}


def node_validate(state: PipelineState) -> PipelineState:
    validator = DataValidator(dataset_name=state.get("dataset_name", "dataset"))
    report = validator.validate(state["df"], key_columns=state.get("key_columns"))
    return {"report": report}


def _fallback_summary(report: ValidationReport) -> str:
    """Deterministic, no-API-key summary. Used when no LLM key is set, so
    the graph is runnable end-to-end for testing/demo purposes. Swap this
    out for the LLM branch in `node_summarize` once you have a key."""
    if not report.findings:
        return f"No data-quality issues found in {report.row_count} rows."
    top = sorted(report.findings, key=lambda f: f.row_count, reverse=True)[:3]
    lines = [
        f"{report.dataset_name}: {report.critical_count} critical, "
        f"{report.warning_count} warning finding(s) across {report.row_count} rows "
        f"(~{report.accuracy_estimate()}% clean-row estimate)."
    ]
    for f in top:
        lines.append(f"  - [{f.severity}] {f.rule} on '{f.column}': {f.message}")
    return "\n".join(lines)


def node_summarize(state: PipelineState) -> PipelineState:
    report = state["report"]
    use_llm = bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))
    if not use_llm:
        return {"summary": _fallback_summary(report)}

    from dataguard.chain import get_chat_model
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    findings_text = "\n".join(
        f"- [{f.severity}] {f.rule} on '{f.column}' ({f.row_count} rows): {f.message}"
        for f in report.findings
    ) or "No findings."

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a data-quality analyst. Summarize these validation "
                "findings for a business stakeholder in 3-5 sentences: call out "
                "the most serious issues first, note overall data health, and "
                "avoid jargon.",
            ),
            ("human", "Dataset: {dataset_name} ({row_count} rows)\n\nFindings:\n{findings}"),
        ]
    )
    chain = prompt | get_chat_model() | StrOutputParser()
    summary = chain.invoke(
        {
            "dataset_name": report.dataset_name,
            "row_count": report.row_count,
            "findings": findings_text,
        }
    )
    return {"summary": summary}


def node_decide_review(state: PipelineState) -> PipelineState:
    report = state["report"]
    return {"needs_human_review": report.critical_count > 0}


def route_after_decide(state: PipelineState) -> str:
    return "human_review" if state.get("needs_human_review") else "report"


def node_human_review(state: PipelineState) -> PipelineState:
    """Placeholder for a real human-in-the-loop step. In a production graph
    this would pause execution (LangGraph supports interrupts) and wait for
    a reviewer's sign-off; here it just attaches a note so the graph shape
    and routing are demonstrable end-to-end without a live reviewer."""
    report = state["report"]
    note = (
        f"Flagged for human review: {report.critical_count} critical finding(s). "
        "A reviewer should confirm these before the data is used downstream."
    )
    return {"human_review_note": note}


def node_report(state: PipelineState) -> PipelineState:
    report = state["report"]
    parts = [
        f"=== DataGuard AI Report: {report.dataset_name} ===",
        f"Rows: {report.row_count} | Columns: {report.column_count}",
        f"Critical: {report.critical_count} | Warning: {report.warning_count}",
        "",
        "Summary:",
        state.get("summary", ""),
    ]
    if state.get("human_review_note"):
        parts += ["", "Human Review:", state["human_review_note"]]
    return {"final_report_text": "\n".join(parts)}


# ---- graph assembly ---------------------------------------------------


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("ingest", node_ingest)
    graph.add_node("validate", node_validate)
    graph.add_node("summarize", node_summarize)
    graph.add_node("decide_review", node_decide_review)
    graph.add_node("human_review", node_human_review)
    graph.add_node("report", node_report)

    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "validate")
    graph.add_edge("validate", "summarize")
    graph.add_edge("summarize", "decide_review")
    graph.add_conditional_edges(
        "decide_review",
        route_after_decide,
        {"human_review": "human_review", "report": "report"},
    )
    graph.add_edge("human_review", "report")
    graph.add_edge("report", END)

    return graph.compile()


def run_pipeline(file_path: str, dataset_name: str = "dataset", key_columns: list[str] | None = None) -> PipelineState:
    app = build_graph()
    return app.invoke(
        {"file_path": file_path, "dataset_name": dataset_name, "key_columns": key_columns or []}
    )


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_orders.csv"
    result = run_pipeline(path, dataset_name=os.path.basename(path), key_columns=["order_id"])
    print(result["final_report_text"])
