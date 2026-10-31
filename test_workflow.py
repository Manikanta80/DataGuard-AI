"""Tests for the LangGraph pipeline. Runs fully offline — with no
OPENAI_API_KEY/ANTHROPIC_API_KEY set, the summarize node uses the
deterministic fallback summary, so the whole graph (including the
conditional human-review branch) is exercisable without any API key."""

import os

import pandas as pd
import pytest

from dataguard.workflow import build_graph, run_pipeline


@pytest.fixture
def clean_csv(tmp_path):
    df = pd.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
    path = tmp_path / "clean.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def dirty_csv(tmp_path):
    df = pd.DataFrame({"id": [1, 1, 3], "val": [10, 20, None]})
    path = tmp_path / "dirty.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_graph_compiles():
    app = build_graph()
    assert app is not None


def test_clean_dataset_skips_human_review(clean_csv, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = run_pipeline(clean_csv, dataset_name="clean", key_columns=["id"])
    assert result["needs_human_review"] is False
    assert "human_review_note" not in result
    assert "DataGuard AI Report" in result["final_report_text"]


def test_dirty_dataset_triggers_human_review(dirty_csv, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = run_pipeline(dirty_csv, dataset_name="dirty", key_columns=["id"])
    assert result["report"].critical_count > 0  # duplicate key on id=1
    assert result["needs_human_review"] is True
    assert "Human Review" in result["final_report_text"]
    assert "human_review_note" in result


def test_final_report_contains_summary_text(clean_csv, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = run_pipeline(clean_csv, dataset_name="clean", key_columns=["id"])
    assert result["summary"] in result["final_report_text"]
