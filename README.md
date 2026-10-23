# policypilot

A governed regulatory/policy Q&A copilot — the first of three portfolio projects
demonstrating agentic AI patterns on Databricks + Azure (Q&A, AI reviewer, generation).
policypilot answers questions about SEC 10-K filings and cites the filing every claim
came from, refusing rather than guessing when it can't ground an answer.

## Status

**Milestone 1 done: local prototype.** Real SEC EDGAR filings, a local Chroma vector
store, an open-weight LLM (`gpt-oss-120b` via Groq), and a Streamlit UI — all runnable on
a laptop with no cloud account.

**Milestone 2 done: deployed and verified on real Azure Databricks.** Workspace, Unity
Catalog, Key Vault, GitHub OIDC federation, Vector Search and a live Databricks App were
all stood up and tested end to end: a deployed question about Apple's supply chain got a
grounded answer with `[n]` citations back to Apple's actual 10-K, and an off-topic
question ("What is the capital of France?") was refused. The Vector Search endpoint was
deleted and the app stopped afterward to stop billing (they bill by the hour; everything
else — workspace, UC catalog/table, Key Vault, OIDC setup, gateway services — stays up at
near-zero cost). See **[docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md)** for the full
step-by-step (including every error hit and how to avoid it).

**Milestone 3 done, except guardrails: broader data, Unity Gateway, CI-gated eval** (see
[docs/UPGRADE_PLAN.md](docs/UPGRADE_PLAN.md)). 20-ticker ingestion, LLM-based
ticker/company detection in `plan_node`, Unity Gateway in front of Groq
(`resources/ai_gateway.yml`: a model provider service + model service, deployed by the
bundle along with their grants), and a Databricks Agent Evaluation gate in `ci.yml` that
has passed against the hosted judges. Guardrails on the model service remain an optional
one-time UI step — see [docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md) Part 7.

The code is structured so that swapping backends is a config change, not a rewrite (see
[Architecture](#architecture) below).

## Architecture

| Layer | Local | Databricks/Azure |
|---|---|---|
| Vector store | `retrieval/local_chroma.py` (Chroma + sentence-transformers) | `retrieval/databricks_vector_search.py` — Delta Sync index `topsecretindex` on endpoint `aiendpoint`, over `topsecretcatalouge.filing.chunks` (self-managed 384-dim embeddings) |
| LLM | `agent/llm.py` `GroqLLMClient`, key from `.env` — guardrail-free by design (dev sandbox) | `DatabricksGatewayLLMClient` calling the `topsecretcatalouge.filing.gpt-oss` Unity Gateway model service (`resources/ai_gateway.yml`), which routes to Groq through a provider service holding the key — no raw Groq key in the App's env, see [docs/UPGRADE_PLAN.md](docs/UPGRADE_PLAN.md) |
| Structured lookup | `ingestion/manifest.py` (local JSON) | UC Function over a Delta table (not built yet) |
| UI | `streamlit run` locally | Databricks App (`resources/apps.yml` + `app.yaml`, deploys the whole repo since the app imports the full `policypilot` package) |
| Ingestion | `python -m policypilot.ingestion.pipeline` | `notebooks/seed_chunks_table.py` (manual, self-contained) |

The catalog, schema, endpoint and index names live in `src/policypilot/config.py`,
`databricks.yml`, `resources/ai_gateway.yml` and `notebooks/seed_chunks_table.py`; change
them together if your workspace uses different names (DEPLOYMENT_GUIDE Part 5.0).

Switch backends via `PP_ENV` (`local` or `databricks`). The agent itself
(`agent/graph.py`, a LangGraph plan → retrieve → answer → verify loop) never changes —
it only talks to the `VectorStore` and `LLMClient` protocols in `retrieval/base.py` and
`agent/llm.py`.

### Deferred: not Mosaic AI yet

Databricks' "Mosaic AI" branding covers Agent Framework (log an agent to Unity Catalog
via MLflow, deploy behind Model Serving), Agent Evaluation, and Agent Bricks. This
project uses Agent Evaluation (CI-gated, see below) and Unity AI Gateway (governance for
the Groq call) but not the rest — the agent itself still runs as plain LangGraph, not
logged to UC or served behind Model Serving. Full Foundation Model migration (swapping
Groq for a Databricks-native model) was deliberately reversed — see
[docs/UPGRADE_PLAN.md](docs/UPGRADE_PLAN.md) item 3 — since a Unity Gateway model service in
front of Groq provides the same guardrails without a provider migration.

The agent is deliberately not a bare RAG chain: `verify_node` is a hard citation gate —
an answer with no `[n]` citation back to retrieved context is replaced with a refusal
before it reaches the user. (gpt-oss sometimes cites as `【1】`; those are normalized to
`[1]` rather than refused.)

## Local setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
uv sync --extra dev
cp .env.example .env
# edit .env: set GROQ_API_KEY, and put your real email in SEC_EDGAR_USER_AGENT
```

**Groq free tier.** It caps requests at 8,000 tokens per minute and 200,000 tokens per
day, shared by everything using the key (local app, CI eval, deployed app). Keep
`PP_RETRIEVAL_K=3` in `.env` (and `app.yaml` sets the same for the deployed app): each
chunk is ~1,500 tokens, and the default of 5 exceeds the per-minute cap.

**Windows notes.**
- If an antivirus inspects HTTPS (e.g. Kaspersky), Python can't reach Groq
  (`APIConnectionError`). `uv pip install truststore` makes Python trust the Windows
  certificate store; `config.py` uses it automatically when installed.
- For GPU embeddings, install the CUDA build of torch with
  `uv pip install torch --index-url https://download.pytorch.org/whl/cu124 --reinstall-package torch`.
  A plain `uv run` re-syncs to the lockfile and removes it (and `truststore`); use
  `uv run --no-sync` after installing either.

### 1. Ingest filings

Pulls the latest 10-K for 20 large-cap tickers (`DEFAULT_TICKERS` in `config.py`) from
SEC EDGAR (public, no auth), chunks them, embeds them locally, and stores them in
`data/chroma/`.

```bash
uv run python -m policypilot.ingestion.pipeline
```

### 2. Run the chat UI

```bash
uv run streamlit run src/policypilot/app/streamlit_app.py
```

### 3. Run the eval harness

Runs golden-set questions (`eval/golden_dataset.jsonl`: 47 questions across the ingested
tickers, including one deliberately off-corpus question that must be refused) through the agent
via Databricks Agent Evaluation (`mlflow.genai.evaluate`). With `DATABRICKS_HOST` set, it
uses the hosted judges (safety, relevance, retrieval groundedness); without it, only the
local citation check runs. It prints a per-question breakdown and exits non-zero if any
mean score falls below its threshold (`THRESHOLDS` in `eval/agent_eval.py`, overridable via
`PP_EVAL_MIN_<SCORER>`) or a check errored.

```bash
uv sync --extra dev --extra eval
PP_EVAL_LIMIT=10 uv run python -m policypilot.eval.agent_eval   # sample 10 + the refusal question
```

The full set uses ~280k tokens, more than Groq's free daily cap, so use `PP_EVAL_LIMIT`.

### Tests

```bash
uv run pytest
uv run ruff check .
```

## CI/CD

- **CI** (`.github/workflows/ci.yml`): ruff + pytest on every push and PR. The
  `agent-eval` job runs on pull requests and manual runs only ("Run workflow"), since each
  run spends ~66k Groq tokens: a fixed 10-question sample plus the refusal question,
  scored by the Databricks judges, failing the build below threshold. Latest run: safety
  1.00, relevance 1.00, citations/refusal 1.00, groundedness 0.80.
- **CD** (`.github/workflows/cd.yml`, manual): OIDC login to Azure, reads the Groq key from
  Key Vault, `databricks bundle deploy -t dev`. The first deploy into a fresh schema fails
  once with "model provider service … does not exist"; re-run it.

Both use GitHub environments (`ci`, `dev`) whose secrets are listed at the top of each
workflow file; no cloud credentials are stored in GitHub.

## Redeploying for a demo

Only the hourly-billed pieces were torn down. To bring the app back
([docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md) Parts 9–11):

1. **Recreate** the Vector Search endpoint `aiendpoint` (Standard) and the Delta Sync
   index `topsecretcatalouge.filing.topsecretindex` (primary key `chunk_id`, existing
   embeddings in `embedding`, dimension 384, Triggered).
2. **Grant the app's own service principal** `SELECT` on the new index, and **Can Use**
   on the endpoint.
3. **Start** the app in Databricks Apps, then **Deploy** from the bundle's
   `.bundle/policypilot/dev/files` folder.
4. Test, then **tear down**: stop the app, delete the index, delete the endpoint.

## Known gaps

- **Filing-date questions** ("When was X's 10-K filed?") are answered and cited, but the
  groundedness judge fails them: the date comes from chunk metadata, which isn't part of
  the context text the judge sees.
- **Guardrails** on the gateway model service aren't configured (optional UI step), and
  nothing tests them automatically — the CI eval calls Groq directly.

## Repo layout

```
src/policypilot/
├── config.py               # env-driven settings, local vs. databricks backend selection
├── ingestion/               # SEC EDGAR fetch -> chunk -> embed -> store
├── retrieval/                # VectorStore protocol + local/Databricks implementations
├── agent/                    # LLM client protocol, tools, LangGraph agent
├── eval/                     # golden dataset + Databricks Agent Evaluation gate
└── app/                       # Streamlit chat UI (becomes the Databricks App)
notebooks/seed_chunks_table.py # Self-contained: fetch+chunk+embed+write into UC, run in-workspace
scripts/setup_unity_catalog.py # UC provisioning script (dev catalog was created by hand instead)
resources/, databricks.yml     # Databricks Asset Bundle (deploy config)
app.yaml, requirements.txt     # Databricks App runtime config (repo root, since apps.yml deploys the whole repo)
.github/workflows/              # CI (lint, tests, eval gate) + CD (manual bundle deploy via OIDC)
screenshots/                    # the local app: a cited answer and its sources
```
