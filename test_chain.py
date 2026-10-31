"""Tests for the LangChain retrieval layer. These only exercise the
no-API-key keyword-retrieval path, so they run offline / in CI without
secrets. The LLM-calling `.ask()` path is exercised manually once you have
an API key (see README)."""

import pandas as pd

from dataguard.validator import DataValidator
from dataguard.chain import DataQAChain, findings_to_documents


def _sample_report():
    df = pd.DataFrame(
        {
            "id": [1, 2, 2],
            "price": [10, None, 9999],
        }
    )
    return DataValidator("t").validate(df, key_columns=["id"], ranges={"price": (0, 100)})


def test_findings_to_documents_count_matches_findings():
    report = _sample_report()
    docs = findings_to_documents(report)
    assert len(docs) == len(report.findings)


def test_findings_to_documents_handles_empty_report():
    df = pd.DataFrame({"a": [1, 2, 3]})
    report = DataValidator("t").validate(df)
    docs = findings_to_documents(report)
    assert len(docs) == 1
    assert "No data-quality issues" in docs[0].page_content


def test_keyword_retrieval_prioritizes_relevant_doc():
    report = _sample_report()
    qa = DataQAChain(report)
    docs = qa.retrieve("price out of range", retrieval="keyword")
    assert any(d.metadata["column"] == "price" for d in docs)


def test_keyword_retrieval_returns_results_even_with_no_term_match():
    report = _sample_report()
    qa = DataQAChain(report)
    docs = qa.retrieve("completely unrelated gibberish query zzz", retrieval="keyword")
    assert len(docs) > 0  # falls back to top-k rather than returning nothing
