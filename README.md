# DataGuard AI

A small, real, end-to-end project combining classic data-quality engineering
with the modern LLM-orchestration stack: **LangChain**, **LangGraph**, and a
hand-written **MCP (Model Context Protocol) server**.

It was built deliberately on top of data-validation work (missing values,
duplicate keys, type mismatches, out-of-range values, statistical outliers)
rather than a generic chatbot demo — the point is to show the same kind of
rigor a QA/data-quality background brings, applied to an AI-native stack.

## What it actually does

1. **`dataguard/validator.py`** — a dependency-free rule engine (Pandas
   only) that runs configurable checks against a DataFrame and returns a
   structured `ValidationReport` of `Finding`s. No API key needed; fully
   unit-tested.
2. **`dataguard/chain.py`** — a LangChain retrieval layer that lets you ask
   natural-language questions about a validation report ("what's wrong with
   the price column?"). Falls back to zero-dependency keyword retrieval when
   no embeddings API key is set, so the retrieval *pattern* is explorable
   without any credentials.
3. **`dataguard/workflow.py`** — a LangGraph graph that orchestrates the
   whole pipeline as a proper multi-step graph, including a **conditional
   edge**: datasets with critical findings get routed to a `human_review`
   node before the final report; clean datasets skip straight to the
   report.
4. **`dataguard/mcp_server.py`** — exposes three tools
   (`validate_dataset`, `get_critical_findings`, `explain_finding`) over the
   Model Context Protocol, so any MCP client (Claude Desktop, another
   agent) can call this project's validation engine directly.

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env        # optional — see "Running without an API key" below
pytest tests/ -v             # 21 tests, all pass with no API key
python -m dataguard.workflow data/sample_orders.csv
```

### Running without an API key

Everything in this repo is runnable and testable with **zero API keys**:
- The validator has no external dependency at all.
- `DataQAChain.ask()` and the LangGraph `summarize` node fall back to
  deterministic, no-API logic when `OPENAI_API_KEY`/`ANTHROPIC_API_KEY`
  aren't set, so the *shape* of the AI pipeline is still fully exercisable.

Set either key in `.env` to switch on real LLM calls for summarization and
semantic Q&A.

### Running the MCP server

```bash
python -m dataguard.mcp_server
```

To use it from Claude Desktop, add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "dataguard": {
      "command": "python",
      "args": ["-m", "dataguard.mcp_server"],
      "cwd": "/absolute/path/to/dataguard-ai"
    }
  }
}
```

Then ask Claude Desktop something like *"Use the dataguard tool to validate
data/sample_orders.csv and tell me what's critical."*

## Project structure

```
dataguard-ai/
├── dataguard/
│   ├── validator.py      # rule engine (no external deps)
│   ├── chain.py          # LangChain Q&A over findings
│   ├── workflow.py       # LangGraph orchestration
│   └── mcp_server.py     # MCP tool server
├── tests/                 # 21 tests, all offline
├── data/sample_orders.csv # intentionally messy sample data
├── requirements.txt
└── .env.example
```

## A known limitation, documented rather than hidden

The z-score outlier check has a real statistical blind spot: a single
extreme value in a small sample inflates the standard deviation enough to
mask its own z-score. It's documented in `validator.py` and covered by a
test (`test_outlier_zscore_single_extreme_value_in_tiny_sample_is_masked`)
that asserts the masking *happens* rather than hiding it — which is also
why that check is marked severity `"info"` rather than `"critical"`.

## Extending it

- Swap `InMemoryVectorStore` in `chain.py` for Chroma/FAISS/pgvector once
  findings volume justifies a real vector store.
- Add more custom rules via `DataValidator.add_rule(name, fn)` — see
  `tests/test_validator.py::test_custom_rule_is_applied` for the pattern.
- The LangGraph `human_review` node is currently a placeholder; LangGraph
  supports real interrupts for pausing on an actual human decision —
  a natural next step.
