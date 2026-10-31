"""LangChain layer: lets a user ask natural-language questions about a
ValidationReport ("which columns have the worst issues?", "summarize the
critical findings") without hand-rolling prompt plumbing.

Design notes for anyone reading this as a learning reference:
- We don't reach for a vector store / embeddings here. A ValidationReport
  for one dataset run is small (tens of findings, not thousands of docs),
  so a vector DB would be over-engineering. Instead we use LangChain's
  `Document` + a simple in-memory `InMemoryVectorStore` so the retrieval
  *pattern* is still real and swappable for a proper vector store later
  (Chroma, FAISS, pgvector...) without changing the calling code.
- The LLM itself is provider-agnostic via `get_chat_model()` below: set
  OPENAI_API_KEY or ANTHROPIC_API_KEY in your environment / .env file and
  it picks whichever is available.
"""

from __future__ import annotations

import os
from typing import Literal

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dataguard.validator import ValidationReport, Finding

SYSTEM_PROMPT = """You are a data-quality analyst assistant. You answer questions \
about a dataset's validation findings using ONLY the context provided below. \
Be concise, specific, and reference column names and counts. If the context \
doesn't contain the answer, say so plainly instead of guessing.

Context (validation findings for dataset '{dataset_name}'):
{context}
"""


def findings_to_documents(report: ValidationReport) -> list[Document]:
    """Turn each Finding into a retrievable Document. One finding = one
    document keeps retrieval granular (a question about one column pulls
    just that column's findings, not the whole report)."""
    docs = []
    for i, f in enumerate(report.findings):
        text = (
            f"Rule: {f.rule}\n"
            f"Severity: {f.severity}\n"
            f"Column: {f.column}\n"
            f"Affected rows: {f.row_count}\n"
            f"Detail: {f.message}"
        )
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "rule": f.rule,
                    "severity": f.severity,
                    "column": f.column,
                    "finding_index": i,
                },
            )
        )
    if not docs:
        docs.append(
            Document(
                page_content="No data-quality issues were found in this dataset.",
                metadata={"rule": "none", "severity": "info", "column": None, "finding_index": -1},
            )
        )
    return docs


def get_chat_model(provider: Literal["auto", "openai", "anthropic"] = "auto", temperature: float = 0.0):
    """Returns a LangChain chat model. Picks whichever provider has an API
    key set, so the same code runs against OpenAI or Anthropic."""
    if provider in ("auto", "openai") and os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)
    if provider in ("auto", "anthropic") and os.getenv("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model="claude-sonnet-4-5", temperature=temperature)
    raise RuntimeError(
        "No LLM API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY "
        "(e.g. in a .env file — see .env.example)."
    )


class DataQAChain:
    """Retrieval-augmented Q&A over a single validation report.

    Usage:
        chain = DataQAChain(report)
        chain.ask("What's the most serious issue in this dataset?")
    """

    def __init__(self, report: ValidationReport, provider: str = "auto"):
        self.report = report
        self.docs = findings_to_documents(report)
        self._provider = provider
        self._vectorstore = None  # built lazily — needs an embeddings call

    def _ensure_vectorstore(self):
        if self._vectorstore is not None:
            return
        from langchain_core.vectorstores import InMemoryVectorStore

        if os.getenv("OPENAI_API_KEY"):
            from langchain_openai import OpenAIEmbeddings

            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        else:
            raise RuntimeError(
                "Semantic retrieval needs an embeddings provider. Set "
                "OPENAI_API_KEY, or call `ask(..., retrieval='keyword')` "
                "to use the no-API-key keyword fallback instead."
            )
        self._vectorstore = InMemoryVectorStore.from_documents(self.docs, embeddings)

    def _keyword_retrieve(self, query: str, k: int = 5) -> list[Document]:
        """A zero-dependency fallback retriever: simple term overlap. Used
        when no embeddings provider is configured, and in tests, so the
        retrieval *flow* is exercisable without any API key."""
        terms = {t.lower() for t in query.split() if len(t) > 2}
        scored = []
        for d in self.docs:
            text = d.page_content.lower()
            score = sum(1 for t in terms if t in text)
            scored.append((score, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:k]] or self.docs[:k]

    def retrieve(self, query: str, k: int = 5, retrieval: str = "auto") -> list[Document]:
        if retrieval == "keyword":
            return self._keyword_retrieve(query, k)
        if retrieval == "auto" and not os.getenv("OPENAI_API_KEY"):
            return self._keyword_retrieve(query, k)
        self._ensure_vectorstore()
        return self._vectorstore.similarity_search(query, k=k)

    def ask(self, question: str, retrieval: str = "auto") -> str:
        relevant_docs = self.retrieve(question, retrieval=retrieval)
        context = "\n\n".join(d.page_content for d in relevant_docs)

        prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", "{question}")]
        )
        llm = get_chat_model(self._provider)
        chain = prompt | llm | StrOutputParser()
        return chain.invoke(
            {
                "dataset_name": self.report.dataset_name,
                "context": context,
                "question": question,
            }
        )
