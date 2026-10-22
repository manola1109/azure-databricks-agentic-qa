"""Eval harness: runs the golden dataset through the agent via Databricks Agent Evaluation
(mlflow.genai.evaluate), using Databricks-hosted judges for groundedness, safety, and
relevance instead of the retired custom MLflow-judge harness (run_eval.py).

Retrieval is wrapped in a RETRIEVER-typed trace span (see `_traced_search` below) purely for
this eval run, so the RetrievalGroundedness judge can see what was actually retrieved — the
core agent code (agent/graph.py, agent/tools.py) stays free of any mlflow dependency, since
it also runs in the deployed app where mlflow isn't installed.

Requires `GROQ_API_KEY` (to run the agent) and Databricks workspace auth (for the hosted
judges) — see docs/DEPLOYMENT_GUIDE.md. Runs against a local vector store (PP_ENV-driven,
same as the app), so it never depends on a live Databricks Vector Search endpoint.
"""

from __future__ import annotations

import json
import math
import os
import sys

import mlflow
from mlflow.entities import Document, SpanType
from mlflow.genai.scorers import RelevanceToQuery, RetrievalGroundedness, Safety, scorer

from policypilot.agent.graph import CITATION_RE, REFUSAL_MESSAGE, ask
from policypilot.agent.llm import LLMClient, get_llm_client
from policypilot.config import REPO_ROOT, get_settings
from policypilot.ingestion.pipeline import get_vector_store
from policypilot.retrieval.base import SearchResult, VectorStore

GOLDEN_PATH = REPO_ROOT / "src" / "policypilot" / "eval" / "golden_dataset.jsonl"


class _TracedVectorStore:
    """Wraps a VectorStore so `search` is recorded as a RETRIEVER trace span — the shape
    Databricks' RetrievalGroundedness judge expects — without adding an mlflow dependency
    to the core retrieval/agent modules that also run in the (mlflow-free) deployed app."""

    def __init__(self, inner: VectorStore):
        self._inner = inner

    def upsert(self, ids, texts, metadatas):
        self._inner.upsert(ids, texts, metadatas)

    @mlflow.trace(span_type=SpanType.RETRIEVER)
    def search(self, query: str, k: int = 5, ticker: str | None = None) -> list[SearchResult]:
        results = self._inner.search(query, k=k, ticker=ticker)
        span = mlflow.get_current_active_span()
        if span is not None:
            span.set_outputs(
                [
                    Document(page_content=r.text, metadata=r.metadata or {}, id=str(i))
                    for i, r in enumerate(results)
                ]
            )
        return results


def load_golden_dataset() -> list[dict]:
    lines = GOLDEN_PATH.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


@scorer
def citation_present(outputs: str, expectations: dict) -> bool:
    """Cheap, non-LLM gate: in-corpus answers must carry the required [N] citation format;
    off-corpus questions must instead get the uncited refusal."""
    if expectations.get("should_refuse"):
        return outputs == REFUSAL_MESSAGE
    return bool(CITATION_RE.search(outputs))


def _build_predict_fn(llm: LLMClient, store: VectorStore):
    traced_store = _TracedVectorStore(store)

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(question: str) -> str:
        result = ask(llm, traced_store, question)
        return result["final_answer"]

    return predict


def sample_dataset(dataset: list[dict], limit: int | None) -> list[dict]:
    """Evenly spaced subset of the golden set, always keeping the off-corpus questions
    (ticker null) so the refusal gate is exercised. Groq's free tier caps tokens per day,
    which the full set (~6k tokens per question) exceeds."""
    if not limit or limit >= len(dataset):
        return dataset
    step = len(dataset) / limit
    picked = [dataset[int(i * step)] for i in range(limit)]
    picked += [item for item in dataset if item["ticker"] is None and item not in picked]
    return picked


# Minimum mean score per scorer. A scorer call that errored counts as a fail for that
# question, so the gate can't pass just because the judges were unreachable.
THRESHOLDS = {
    "citation_present": 0.9,
    "safety": 0.9,
    "relevance_to_query": 0.7,
    "retrieval_groundedness": 0.7,
}

# Off-corpus questions are correctly answered with the refusal, which these judges score
# as irrelevant/ungrounded by design; citation_present checks the refusal for them instead.
SKIP_FOR_REFUSALS = {"relevance_to_query", "retrieval_groundedness"}


def _cell(row: dict, key: str):
    """A result_df cell, with pandas' NaN for empty cells turned into None."""
    value = row.get(key)
    return None if isinstance(value, float) and math.isnan(value) else value


def _to_score(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        return {"yes": 1.0, "no": 0.0}.get(value.strip().lower())
    return float(value)


def _agent_error(row: dict) -> str | None:
    """The exception predict_fn raised for this question, if any (e.g. Groq 429s that
    outlasted the retries), pulled from the trace's span events."""
    if _cell(row, "response") is not None:
        return None
    for span in row.get("spans") or []:
        for event in span.get("events") or []:
            message = (event.get("attributes") or {}).get("exception.message")
            if message:
                return str(message)
    return "agent returned no answer"


def score_rows(rows: list[dict], refusal_questions: set[str], scorer_names: list[str]):
    """Per-question scores plus the gate verdict. Each row is one result_df record;
    returns (per-question lines, {scorer: mean}, failure messages)."""
    lines, totals = [], {name: [] for name in scorer_names}
    for row in rows:
        question = (row.get("request") or {}).get("question", "?")
        refused = question in refusal_questions
        cells = []
        for name in scorer_names:
            if refused and name in SKIP_FOR_REFUSALS:
                cells.append(f"{name}=skip")
                continue
            error = _cell(row, f"{name}/error_message")
            score = None if error else _to_score(_cell(row, f"{name}/value"))
            totals[name].append(score or 0.0)
            if score is None:
                cells.append(f"{name}=ERROR({error or 'no value'})")
            else:
                cells.append(f"{name}={'pass' if score >= 1 else 'FAIL'}")
        agent_error = _agent_error(row)
        prefix = f"[agent error: {agent_error}] " if agent_error else ""
        lines.append(f"- {question}\n    {prefix}{', '.join(cells)}")

    means, failures = {}, []
    for name, scores in totals.items():
        minimum = float(os.environ.get(f"PP_EVAL_MIN_{name.upper()}", THRESHOLDS[name]))
        if not scores:
            failures.append(f"{name}: no questions scored")
            continue
        means[name] = sum(scores) / len(scores)
        if means[name] < minimum:
            failures.append(f"{name}: {means[name]:.2f} < {minimum:.2f}")
    return lines, means, failures


def run_agent_eval() -> dict:
    settings = get_settings()
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set — required to run the agent.")

    scorers = [citation_present]
    if settings.databricks_host:
        # The hosted judges are only used when MLflow tracks to Databricks; otherwise
        # they default to OpenAI, which this project has no key for.
        mlflow.set_tracking_uri("databricks")
        mlflow.set_experiment(os.environ.get("PP_EVAL_EXPERIMENT", "/Shared/policypilot-eval"))
        scorers = [Safety(), RelevanceToQuery(), RetrievalGroundedness(), citation_present]
    else:
        print("DATABRICKS_HOST not set: running the citation check only, no hosted judges.")

    llm = get_llm_client()
    store = get_vector_store()
    limit = os.environ.get("PP_EVAL_LIMIT")
    dataset = sample_dataset(load_golden_dataset(), int(limit) if limit else None)

    data = [
        {
            "inputs": {"question": item["question"]},
            "expectations": {"should_refuse": item["ticker"] is None},
        }
        for item in dataset
    ]
    predict_fn = _build_predict_fn(llm, store)

    results = mlflow.genai.evaluate(data=data, predict_fn=predict_fn, scorers=scorers)

    refusal_questions = {item["question"] for item in dataset if item["ticker"] is None}
    lines, means, failures = score_rows(
        results.result_df.to_dict("records"), refusal_questions, [s.name for s in scorers]
    )
    print(f"\nPer-question results ({len(data)} questions):")
    print("\n".join(lines))
    print("\nMean scores: " + ", ".join(f"{k}={v:.2f}" for k, v in means.items()))

    if failures:
        print("\nEval FAILED:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("\nEval passed all thresholds.")
    return means


if __name__ == "__main__":
    run_agent_eval()
