# ForgeTune

A local-first fine-tuning studio for instruction and completion datasets. The browser UI talks
to a small Node server, which proxies to a FastAPI/LangGraph trainer running on an NVIDIA GPU —
either on your own machine or on a remote GPU machine you reach over SSH.

## How it runs

```
Browser ──► Node web server (server.mjs, :4173) ──► Trainer API (trainer/app.py, FastAPI + LangGraph)
                                                             │
                                                             ├──► local NVIDIA GPU (:8000), or
                                                             ├──► remote GPU machine (:18000 via SSH tunnel)
                                                             └──► OpenAI-compatible API (optional, hosted)
                                                                      │
                                         runtime data: datasets, runs, adapters, workflows
```

- **Web server** (`server.mjs`, `public/`, `index.html`): serves the UI and proxies `/api/*` to the trainer; it holds no API keys.
- **Trainer** (`trainer/`): dataset import and audit, LangGraph workflow, training, and evaluation.
- **Runtime data** (`trainer/data/`, `.forge-data/`): never committed.

## Requirements

| Component | Needed for | Version |
|---|---|---|
| [Node.js](https://nodejs.org/) | The web UI | 20.12 or newer |
| Python | The trainer | 3.11 or newer |
| [PyTorch with CUDA](https://pytorch.org/get-started/locally/) | Training and model inference | Matching your CUDA driver |
| NVIDIA GPU | Training and model inference | 24 GB of memory (the model catalog and launcher are sized for it) |

The UI and CPU-only dataset analysis run without a GPU. PyTorch is not listed in
`trainer/requirements.txt` because the right build depends on your CUDA version; install it
first from the PyTorch site.

## Quick start

### 1. Clone and start the UI

```bash
git clone https://github.com/st101cc/forge_tune.git
cd forge_tune
npm start
```

Open http://127.0.0.1:4173. Without a trainer the UI runs in preview mode: you can look around,
but nothing is analysed or trained. There are no npm dependencies to install.

### 2. Start the trainer

Pick the option that matches your setup. In every case the UI expects the trainer at
`http://127.0.0.1:8000` unless you set `TRAINER_URL` (see [Configuration](#configuration)).

**Linux, or WSL2 on Windows**

```bash
cd trainer
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # pick the URL for your CUDA version
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8000
```

Afterwards `./start.sh` in `trainer/` repeats the install and starts the API. Keep it running while
you use the UI. With WSL2, run `npm start` in Windows or in WSL — localhost is shared either way.

**Windows (native), all in one step**

Create a virtual environment at the repository root with CUDA-enabled PyTorch and the trainer
requirements:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\pip install -r trainer\requirements.txt
```

Then stop any `npm start` server and run:

```powershell
powershell -ExecutionPolicy Bypass -File start-auto.ps1
```

The launcher checks for a local NVIDIA GPU with at least 24 GB, starts the trainer on port 8000,
then starts the UI at http://127.0.0.1:4173. Logs go to `.forge-data/logs/`. Use `-CheckOnly` to
see which target it would pick and `-SelfTest` to check the GPU selection rule.

**Remote GPU machine over SSH**

If your own machine has no suitable GPU, run the trainer on a remote Linux GPU machine and tunnel to it:

1. On the remote machine, clone this repository, create a Python 3.11+ virtual environment at
   the repository root (`.venv`) with CUDA PyTorch and `trainer/requirements.txt`, then run
   `trainer/start-vm-backend.sh`. It starts the API on the remote machine's loopback port 18000
   and logs to `data/logs/backend.log`. No public port is opened.
2. Set up an SSH host alias for the machine in `~/.ssh/config` with key-based login.
3. On your Windows machine, run:

   ```powershell
   powershell -ExecutionPolicy Bypass -File start-auto.ps1 -VmHost <your-ssh-alias>
   ```

   The launcher still prefers a local 24 GB GPU if it finds one. Otherwise it checks that the remote
   API answers, opens an SSH tunnel to port 18000, and starts the UI.

On Linux or macOS, open the tunnel yourself and point the UI at it:

```bash
ssh -N -L 127.0.0.1:18000:127.0.0.1:18000 <your-ssh-alias> &
TRAINER_URL=http://127.0.0.1:18000 npm start
```

To restart the remote API, stop its `uvicorn app:app` process first; the script exits if port 18000
already answers. The launcher does not copy datasets or checkpoints between machines.

## Configuration

Settings are read from environment variables, or from a `.env` file at the repository root
(web server) and `trainer/.env` (trainer). Copy `trainer/.env.example` to start. Never commit `.env` files.

| Variable | Used by | Default | Purpose |
|---|---|---|---|
| `PORT` | Web server | `4173` | UI port |
| `TRAINER_URL` | Web server | `http://127.0.0.1:8000` | Where the UI sends `/api/*` requests |
| `FORGETUNE_DATA_ROOT` | Trainer | `trainer/data` | Where datasets, runs, adapters and workflows are stored |
| `EVALUATOR_MODEL` | Trainer | `Qwen/Qwen2.5-7B-Instruct` | Local model that grades answers (must already be downloaded) |
| `ADVISOR_BASE_URL`, `ADVISOR_API_KEY`, `ADVISOR_MODEL` | Trainer | unset | Optional OpenAI-compatible API for cloud dataset assistance |

Everything works without the `ADVISOR_*` values. See [Optional cloud assistance](#optional-cloud-assistance).

## Workflow

Import data (file, GitHub, Hugging Face) → analyse and benchmark → approve and train (LoRA/QLoRA) →
evaluate base vs tuned, all from the Workspace.

Nothing trains or uses the GPU until you approve it, and no adapter is ever deployed automatically.

## Tests

Test files (`test/`, `test-*.mjs`, `trainer/test_*.py`, `trainer/conftest.py`) are not published in
this repository. If you have them:

```bash
python -m pytest trainer -q
```

Python tests need the trainer dependencies. Browser tests (`node test/test-*.cjs`) need the web
server on :4173 and Playwright (set `PLAYWRIGHT_MODULE` when it is not installed in the project).

## Feature reference

### What is included

- Guided dataset, model/method, training, and evaluation/export workflow
- CSV, JSONL, Parquet, GitHub and Hugging Face dataset entry points
- PEFT, LoRA, and memory-efficient 4-bit QLoRA selections
- 3B, 7B, and guarded 14B model choices for a 24 GB GPU
- A one-job GPU queue, persisted run metadata/logs, cancellation, and adapter artifacts
- Ollama Modelfile generation alongside adapters and evaluation metrics
- LangGraph workflow orchestration with resumable approvals, shared evaluation splits, bounded retries, and candidate comparison

### Dataset schemas

The trainer accepts:

- messages: chat messages compatible with the model chat template
- prompt + completion: paired supervised examples
- text: complete text examples

When no test dataset is supplied, the trainer creates a reproducible 90/10 train/evaluation split using seed 42.

### Dataset-first workspace

Upload CSV, JSONL, or Parquet on the workspace to start a LangGraph analysis-only
run. Local profiling checks the schema and up to 600 rows for quality, duplicates,
language, and likely task. Uncertain intent pauses for a task and explanation;
optional cloud assistance can suggest an interpretation but cannot confirm it for
you. Blocking data issues require replacement or explicitly approved cleanup.

Analysis stops at **Dataset ready** before any baseline inference or training.
**Update dataset** uploads a new file and starts a new analysis linked to its
previous version. Earlier files and clarification records remain available in
version history. Updates replace the file for a new version; they do not merge
rows or edit the prior version.

SQLite checkpoints preserve pending questions across restarts. CPU-only dataset
analysis needs only FastAPI, uvicorn, datasets, python-dotenv, python-multipart,
langgraph, langgraph-checkpoint-sqlite, and aiosqlite; no GPU model is loaded
until you explicitly continue to model evaluation.

### Import from GitHub

Choose **GitHub** in the dataset import panel and paste a public file URL such as
`https://github.com/owner/repo/blob/main/data.jsonl`, or its
`raw.githubusercontent.com` URL. CSV, JSONL, and Parquet files up to 50 MB are
supported. Repository/folder links and private repositories are not supported.

`POST /datasets/github` saves a new local copy and retains its canonical source
URL. The UI then starts the same analysis-only workflow used for file uploads.
Imports never execute repository code, do not follow redirects, and remove partial
downloads on failure.

### Hugging Face Hub imports

Choose **Hugging Face**, paste a dataset page URL or `owner/dataset` ID, and select
its split (normally `train`). **Check dataset** first shows an overview from Hub metadata,
without downloading any data: each subset's row count (estimated for large subsets), its
download size, and the trainer's free disk space. It then preselects a recommended import
scope: **Full dataset** when the split has at most 50,000 rows and uses under half the free
disk, otherwise **Small sample** at 600 rows per subset. You can change the scope before
importing. `GET /datasets/huggingface/overview?url=…&split=…` returns the same data. **Full dataset (all subsets)** is the default:
**Import Full Data** downloads every configuration/subset for that split and
automatically audits each one. All snapshots use the same pinned Hub revision.

After every subset has been processed, **Choose subsets for training** shows
each audit and any download/analysis failures. Select one or more analyzed
subsets, then **Continue with selected subsets**. Each gets a separate saved
workflow; different schemas are never silently merged. Selection itself starts no
GPU work. A subset without the requested split is reported as failed rather than
silently substituted with test or validation data.

Full imports run in the background with row/byte progress and cancellation.
Reloading the workspace restores the current import in the same browser. Each
subset receives a local quality audit of up to 600 rows; sampled audits do not
certify every row. Large files need extra time and disk space for the Arrow cache
and splits.

**Small sample** retains the first 30–10000 rows (default 600) of each subset, capped at
50 MiB per subset. A dataset with one subset goes straight to analysis; with several, every
subset is sampled and audited first, then chosen in the same **Choose subsets for training**
step. A prefix sample is not necessarily representative of all domains.

The importer reserves 1 GiB of free space plus an estimate for the later dataset
cache. Disk exhaustion, network errors, and cancellation remove unfinished files.
A trainer restart marks running imports interrupted; restart those from the
beginning. Completed subset snapshots are retained.

API: `POST /datasets/huggingface` with `mode:"all"` returns HTTP 202 and an import ID.
Poll `GET /datasets/imports/{id}`; cancel with `POST /datasets/imports/{id}/cancel`.
After completion, post `{"configurations":["chat","math"]}` to
`/datasets/imports/{id}/select` to prepare separate workflows. Only one full import
runs per trainer process.

Model inference truncates inputs to 4096 tokens, so very long-context datasets may
need a smaller row count or shorter examples.

### Guided LangGraph workflow

When the trainer is online, the workspace starts a guided workflow that profiles the
dataset, creates a reproducible held-out split, runs base-model baselines, and pauses
before GPU training so the proposed candidates and methods can be reviewed. It then
trains eligible candidates serially, retries failed evaluations at most twice, and
saves base-vs-tuned comparisons plus a final candidate leaderboard under
`<data root>/workflows/`.

The workflow API is available at `/workflow/runs`, `/workflow/runs/{id}`, and
`/workflow/runs/{id}/resume`. The `/runs` endpoints remain available for single-model runs.

### Pipeline safeguards

New workflows ask for an intended task and success metric before model
benchmarking. These are a recorded experiment brief, not an automatically
validated custom scoring function. Evaluation size is configurable from 1 to
200 examples per phase (default 50, bounded by available rows). Under 30
examples is explicitly exploratory.

Dataset audits sample up to 600 rows reproducibly across the entire dataset,
not just its prefix. Privacy findings retain original row IDs. Splits keep
normalized identical inputs together even when answers differ; external test
overlap is rejected. Existing manifests are checked again before training.
This is exact-input protection, not semantic near-duplicate detection.

SFT uses explicit response masks: prompt/completion and classification learn
the target; chat learns the final assistant turn, with earlier turns as context.
Plain text uses full-sequence language modeling. Sequences truncated to zero
target tokens fail clearly. Final evaluation is report-only: poor scores or judge
failures never trigger retraining.

The offline `trainer/pipeline_pilot.py` exercises ticket classification, JSON
extraction and summarization with synthetic fixtures, a cached local model,
and a hard timeout. Use `--seconds` to reduce the default 1,800-second budget
and select an idle GPU via `CUDA_VISIBLE_DEVICES`.

### Prompt review and automatic model analysis

After a dataset is ready, **Test base model & prompts** runs the base model on fixed
development examples. New workflows also automatically test the baseline and two
task-specific prompt strategies. Try improved instructions or reference documents
in the same workflow, inspect the recorded responses, and select a prompt version.
Choose **Keep base model** to finish without training, or **Review a small fine-tune**
to open the training approval. Each workflow allows up to 10 prompt trials.

Development examples stay separate from training and final evaluation. Base and
tuned models use the selected instructions/context on the same final examples.
Label accuracy (with precision, recall, F1, and errors) and structured JSON exact
match are measured where supported. Word overlap is a diagnostic only. A fine-tune
is proposed for review only with at least a 5 percentage point score gain on the
same fully scored held-out examples without format regression.

Reference uploads accept text or Markdown documents; retrieval is per question.
`trainer/quality_probe.py` provides a separate small-corpus RAG experiment using
`validation/rag-corpus.json`. The workflow generates up to 512 tokens per answer and
records whether that limit was reached.

### Development benchmark model selection

New workflows benchmark every GPU-eligible catalog model before prompt experiments.
All candidates receive the same development examples, instruction and 256-token
output cap; a local input over 4,096 tokens is rejected instead of truncated. No
final test questions are used to choose the model.

Selection requires at least two completed models, full correctness coverage and the
same scoring metric. The highest score defines a shortlist within two percentage
points; lower cost, then lower latency, breaks ties. Local models count as free.
This is a provisional small-sample policy, not a statistical significance claim.

**Benchmark models on this dataset** creates a new workflow using the same dataset
and split seed. Use a fresh, untouched final set before claiming a new improvement.

### Automatic task evaluation

`trainer/evaluator.py` grades recorded answers during benchmarking, prompt/RAG trials
and final comparison. It chooses a rubric per question (math, code, summarization,
writing, factual knowledge, or general assistance). Structured targets keep their
deterministic metrics; other answers receive a reference-guided pass/fail/abstain
verdict, with the candidate model's identity hidden.

The default judge is `Qwen/Qwen2.5-7B-Instruct`, loaded in 4-bit mode on the trainer
GPU. Download it once (for example with `huggingface-cli download Qwen/Qwen2.5-7B-Instruct`)
or set `EVALUATOR_MODEL` to another downloaded model; evaluation never downloads
models automatically and sends no data to a hosted API. Code grading is static
review, not execution. Model grades are shown as **estimated correctness**; calibrate
them with human review before deployment.

For completed workflows, **Run automatic evaluation** grades saved base/tuned answers
without retraining and preserves the original in `before-automatic-evaluation.json`.

### Optional cloud assistance

ForgeTune can ask an OpenAI-compatible API for a second opinion when local dataset
profiling is uncertain. Set `ADVISOR_BASE_URL`, `ADVISOR_API_KEY` and `ADVISOR_MODEL`
in `trainer/.env`. The key never reaches the browser, and at most 20 locally redacted
sample rows are sent, only after you request cloud assistance. `GET /advisor/status`
reports whether it is configured. The local auditor remains the authoritative source
for dataset quality and GPU feasibility checks.

Hosted-model features — the AI model picker and benchmarking hosted models alongside
local ones (`trainer/compass_client.py`, `trainer/model_picker.py`) — currently work
only with the specific gateway hard-coded in `compass_client.py`. With any other
endpoint they stay off and ForgeTune uses its built-in model list.

### Ollama

After converting the merged model to GGUF, use the generated Modelfile:

    ollama create forge-support -f Modelfile
    ollama run forge-support

Install [Ollama](https://ollama.com/) first.

The tested trainer uses TRL 0.22 and Transformers 4.x. `trainer/requirements-accelerated.txt`
adds the optional Unsloth QLoRA fast path; ForgeTune falls back to Transformers + TRL + PEFT
when it is not installed.
