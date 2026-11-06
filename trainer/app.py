"""WSL-local API for ForgeTune. Run with ./start.sh from this folder."""
from __future__ import annotations

import asyncio
import hmac
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import urllib.request
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import truststore
# Trust the OS certificate store so downloads work behind TLS-inspecting proxies and antivirus.
truststore.inject_into_ssl()

from fastapi import BackgroundTasks, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from advisor import baseline_compare, cloud_clarify, cloud_status, connection_test, load_source, local_profile, rank_candidates
from graph import WorkflowServices, build_workflow_graph, isolated_inference, accuracy_agent_review, _compare_results
from evaluator import DEFAULT_MODEL, evaluate_results
import compass_client
from langgraph.types import Command

try:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
except ImportError:  # pragma: no cover - requirements install the SQLite saver in production.
    AsyncSqliteSaver = None
from langgraph.checkpoint.memory import MemorySaver

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
load_dotenv(Path.cwd() / ".env")
DATA_ROOT = Path(os.environ.get("FORGETUNE_DATA_ROOT", str(ROOT / "data")))
RUNS_ROOT = DATA_ROOT / "runs"
DATASETS_ROOT = DATA_ROOT / "datasets"
WORKFLOWS_ROOT = DATA_ROOT / "workflows"
WORKFLOW_DB = DATA_ROOT / "workflows.sqlite"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)
DATASETS_ROOT.mkdir(parents=True, exist_ok=True)
WORKFLOWS_ROOT.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="ForgeTune Local Trainer", version="0.1.0")
queue_lock = asyncio.Lock()
active_processes: dict[str, asyncio.subprocess.Process] = {}
active_workflows: dict[str, asyncio.Task] = {}
active_evaluations: dict[str, asyncio.Task] = {}
active_imports: dict[str, tuple[asyncio.Task, threading.Event]] = {}


class DatasetRequest(BaseModel):
    source: Literal["upload", "huggingface"]
    name: str = Field(min_length=1)
    extension: str | None = None
    split: str = "train"
    displayName: str | None = Field(default=None, max_length=255)
    sourceUrl: str | None = Field(default=None, max_length=2048)
    importedRows: int | None = None
    hubConfig: str | None = None
    hubSplit: str | None = None
    hubRevision: str | None = None
    sampleOnly: bool = False
    qualityVersionId: str | None = None


class Hyperparameters(BaseModel):
    rank: int = Field(default=16, ge=1, le=128)
    alpha: int = Field(default=32, ge=1, le=256)
    learning_rate: float = Field(default=2e-4, gt=0, le=1e-2)
    epochs: int = Field(default=3, ge=1, le=10)
    batch_size: int = Field(default=1, ge=1, le=16)
    gradient_accumulation: int = Field(default=8, ge=1, le=128)
    max_sequence_length: int = Field(default=2048, ge=128, le=8192)


class RunRequest(BaseModel):
    dataset: DatasetRequest
    base_model: str
    parameter_method: Literal["lora", "qlora"]
    hyperparameters: Hyperparameters = Field(default_factory=Hyperparameters)
    test_dataset: DatasetRequest | None = None
    split_seed: int = 42
    workflow_id: str | None = None
    candidate_id: str | None = None
    attempt: int = 0
    idempotency_key: str | None = None
    split_manifest_id: str | None = None
    split_manifest_path: str | None = None


class WorkflowRequest(BaseModel):
    analysis_limit: int = Field(default=50, ge=1, le=200)
    task_description: str = Field(default="", max_length=2000)
    success_metric: str = Field(default="", max_length=2000)
    knowledge_documents: list[dict] = Field(default_factory=list, max_length=50)
    auto_analysis: bool = True
    analysis_only: bool = False
    previous_workflow_id: uuid.UUID | None = None
    dataset: DatasetRequest
    goal: Literal["speed", "balanced", "quality"] = "balanced"
    cloud_assist: bool = False
    test_dataset: DatasetRequest | None = None
    split_seed: int = 42
    candidate_limit: int = Field(default=3, ge=1, le=3)
    max_retries: int = Field(default=2, ge=0, le=2)


class WorkflowResumeRequest(BaseModel):
    response: dict


class AnalysisRequest(BaseModel):
    dataset: DatasetRequest
    goal: Literal["speed", "balanced", "quality"] = "balanced"
    cloud_assist: bool = False


class BaselineRequest(BaseModel):
    dataset: DatasetRequest
    goal: Literal["speed", "balanced", "quality"] = "balanced"
    candidate_ids: list[str] = Field(default_factory=list, max_length=3)


class ExperimentExample(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    completion: str = Field(min_length=1, max_length=4000)


class ExperimentRequest(BaseModel):
    use_case: str = Field(min_length=1, max_length=2000)
    success_criteria: str = Field(min_length=1, max_length=2000)
    model_id: str = Field(min_length=1, max_length=200)
    instruction: str = Field(default="", max_length=4000)
    context: str = Field(default="", max_length=8000)
    examples: list[ExperimentExample] = Field(min_length=1, max_length=10)


@app.post("/experiments")
async def run_experiment(request: ExperimentRequest) -> dict:
    from advisor import MODELS
    from datasets import Dataset
    model = next((item for item in MODELS if item["id"] == request.model_id), None)
    if model is None:
        raise HTTPException(422, "Choose a model from the catalog.")
    if not request.use_case.strip() or not request.success_criteria.strip() or any(not item.prompt.strip() or not item.completion.strip() for item in request.examples):
        raise HTTPException(422, "Describe the task, success criteria, and each input and ideal answer.")
    if queue_lock.locked() or active_workflows:
        raise HTTPException(409, "Wait for the current GPU workflow to finish.")
    if gpu_status()["memoryGb"] < model["vram"]:
        raise HTTPException(422, "The selected model exceeds available GPU capacity, or no GPU is connected.")
    instruction = request.instruction
    if request.context.strip():
        instruction += "\n\nReference context:\n" + request.context
    async with queue_lock:
        try:
            results = await asyncio.to_thread(baseline_compare, Dataset.from_list([item.model_dump() for item in request.examples]), [{"model": model, "method": "qlora"}], len(request.examples), instruction=instruction)
        except Exception as error:
            raise HTTPException(503, str(error)[:300]) from error
    result = results[0]
    if result.get("status") != "complete":
        raise HTTPException(503, result.get("message", "Local inference was unavailable."))
    record = {"id": str(uuid.uuid4()), "createdAt": datetime.now(timezone.utc).isoformat(), "request": request.model_dump(), "result": result}
    directory = DATA_ROOT / "experiments"
    directory.mkdir(exist_ok=True)
    (directory / (record["id"] + ".json")).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def run_path(run_id: str) -> Path:
    return RUNS_ROOT / run_id / "run.json"


def read_run(run_id: str) -> dict:
    path = run_path(run_id)
    if not path.exists():
        raise HTTPException(404, "Training run not found")
    return json.loads(path.read_text(encoding="utf-8"))


def write_run(record: dict) -> None:
    path = run_path(record["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def gpu_status() -> dict:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True,
            timeout=5,
        ).strip().splitlines()[0]
        name, memory = [part.strip() for part in output.split(",", 1)]
        memory_gb = round(float(memory.split()[0]) / 1024)
        return {"name": name, "memoryGb": memory_gb, "trainerOnline": True}
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, IndexError):
        return {"name": "NVIDIA GPU unavailable", "memoryGb": 0, "trainerOnline": True}


@app.get("/hardware")
async def hardware() -> dict:
    return gpu_status()


@app.get("/models")
async def model_catalog() -> list[dict]:
    return [
        {"id": "Qwen/Qwen2.5-3B-Instruct", "sizeB": 3, "recommendedVramGb": 8, "methods": ["lora", "qlora"]},
        {"id": "Qwen/Qwen2.5-7B-Instruct", "sizeB": 7, "recommendedVramGb": 13, "methods": ["lora", "qlora"]},
        {"id": "Qwen/Qwen2.5-Coder-7B-Instruct", "sizeB": 7, "recommendedVramGb": 13, "methods": ["lora", "qlora"]},
        {"id": "Qwen/Qwen2.5-14B-Instruct", "sizeB": 14, "recommendedVramGb": 22, "methods": ["qlora"]},
    ]


@app.post("/datasets/validate")
async def validate_dataset(request: DatasetRequest) -> dict:
    if request.source == "upload":
        extension = (request.extension or Path(request.name).suffix.lstrip(".")).lower()
        if extension not in {"csv", "jsonl", "parquet"}:
            raise HTTPException(422, "Supported file types are CSV, JSONL, and Parquet.")
        return {"valid": True, "format": extension, "schema": "pending file upload", "suggestedSplit": "90/10"}
    if "/" not in request.name:
        raise HTTPException(422, "Use a Hugging Face dataset ID such as owner/dataset.")
    return {"valid": True, "format": "huggingface", "schema": "inspect on import", "suggestedSplit": "90/10"}


@app.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)) -> dict:
    extension = Path(file.filename or "").suffix.lower().lstrip(".")
    if extension not in {"csv", "jsonl", "parquet"}:
        raise HTTPException(422, "Supported file types are CSV, JSONL, and Parquet.")
    destination = DATASETS_ROOT / (str(uuid.uuid4()) + "." + extension)
    with destination.open("wb") as output:
        shutil.copyfileobj(file.file, output)
    return {
        "source": "upload",
        "name": str(destination),
        "displayName": file.filename,
        "extension": extension,
        "format": extension,
        "suggestedSplit": "90/10",
    }


class HuggingFaceRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)
    configuration: str | None = Field(default=None, max_length=200)
    split: str = Field(default="train", min_length=1, max_length=100)
    max_rows: int = Field(default=600, ge=30, le=10000)
    mode: Literal["sample", "full", "all"] = "sample"


def write_import(record: dict) -> None:
    directory = DATASETS_ROOT / ".imports"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (str(uuid.UUID(record["id"])) + ".json")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def read_import(import_id: uuid.UUID) -> dict:
    path = DATASETS_ROOT / ".imports" / (str(import_id) + ".json")
    if not path.exists():
        raise HTTPException(404, "Dataset import not found.")
    return json.loads(path.read_text(encoding="utf-8"))


async def execute_import(record: dict, request: HuggingFaceRequest, stop: threading.Event) -> None:
    from hf_import import import_huggingface, import_all_subsets
    def progress(values):
        record.update(values)
        write_import(record)
    def run():
        try:
            if request.mode in {"all", "sample"}:
                record["result"] = import_all_subsets(request.url, request.split, DATASETS_ROOT,
                                                    progress=progress, cancelled=stop.is_set, import_id=record["id"],
                                                    max_rows=request.max_rows if request.mode == "sample" else None)
            else:
                record["result"] = import_huggingface(request.url, request.configuration, request.split, None, DATASETS_ROOT,
                                                     progress=progress, cancelled=stop.is_set, import_id=record["id"])
            record["status"] = "needs_configuration" if record["result"].get("requiresConfiguration") else "complete"
        except InterruptedError as error:
            record.update(status="cancelled", error=str(error))
        except Exception as error:
            record.update(status="failed", error="Hugging Face import failed: " + str(error)[:500])
        record["completedAt"] = datetime.now(timezone.utc).isoformat()
        write_import(record)
    try:
        await asyncio.to_thread(run)
    finally:
        active_imports.pop(record["id"], None)


@app.get("/datasets/imports/{import_id}")
async def get_dataset_import(import_id: uuid.UUID) -> dict:
    record = read_import(import_id)
    active = active_imports.get(str(import_id))
    if active and active[1].is_set() and record["status"] == "running":
        return {**record, "status": "cancelling"}
    return record


@app.post("/datasets/imports/{import_id}/cancel", status_code=202)
async def cancel_dataset_import(import_id: uuid.UUID) -> dict:
    record = read_import(import_id)
    active = active_imports.get(str(import_id))
    if active and record["status"] == "running":
        active[1].set()
        return {**record, "status": "cancelling"}
    return record


@app.get("/datasets/huggingface/overview")
async def huggingface_overview(url: str, split: str = "train") -> dict:
    from hf_import import dataset_overview
    if not url.strip() or len(url) > 2048 or not split.strip() or len(split) > 100:
        raise HTTPException(422, "Enter a dataset URL or ID and a split.")
    try:
        return await asyncio.to_thread(dataset_overview, url, split.strip(), DATASETS_ROOT)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    except Exception as error:
        raise HTTPException(422, "Could not read this dataset from Hugging Face: " + str(error)[:500]) from error


@app.post("/datasets/huggingface")
async def huggingface_dataset(request: HuggingFaceRequest) -> dict:
    from hf_import import import_huggingface
    if request.mode == "sample":
        try:
            result = await asyncio.to_thread(import_huggingface, request.url, request.configuration, request.split, request.max_rows, DATASETS_ROOT)
        except Exception as error:
            raise HTTPException(422, "Hugging Face import failed: " + str(error)[:500]) from error
        if not result.get("requiresConfiguration"):
            return result
        # Several subsets: sample and audit each one in the background, then the user chooses.
    else:
        from hf_import import dataset_id
        try:
            dataset_id(request.url)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
    # ponytail: one background import per trainer process; use a durable queue for multi-worker deployments.
    if active_imports:
        raise HTTPException(409, "Another dataset import is running. Wait for it or cancel it first.")
    record = {"id": str(uuid.uuid4()), "status": "running", "createdAt": datetime.now(timezone.utc).isoformat(),
              "request": request.model_dump(), "importedRows": 0, "importedBytes": 0, "totalRows": None}
    write_import(record)
    stop = threading.Event()
    active_imports[record["id"]] = (asyncio.create_task(execute_import(record, request, stop)), stop)
    return JSONResponse(record, status_code=202)


class SubsetSelectionRequest(BaseModel):
    configurations: list[str | None] = Field(min_length=1)
    previous_workflow_id: uuid.UUID | None = None
    cloud_assist: bool = False


@app.post("/datasets/imports/{import_id}/select")
async def select_import_subsets(import_id: uuid.UUID, request: SubsetSelectionRequest) -> dict:
    record = read_import(import_id)
    if record["status"] != "complete" or not record.get("result", {}).get("allSubsets"):
        raise HTTPException(409, "Wait until all subset audits finish before selecting training data.")
    entries = {item["configuration"]:item for item in record["result"]["subsets"]}
    selected = request.configurations
    if len(set(selected)) != len(selected) or any(name not in entries or entries[name]["status"] != "analyzed" for name in selected):
        raise HTTPException(422, "Choose distinct subsets with completed analysis.")
    if request.previous_workflow_id:
        read_workflow(str(request.previous_workflow_id))
    links = record.setdefault("selectedWorkflows", {})
    workflows = []
    for name in selected:
        key = json.dumps(name)
        if key not in links:
            workflow = await create_workflow(WorkflowRequest(dataset=DatasetRequest(**entries[name]["dataset"]),
                analysis_only=True, auto_analysis=False, candidate_limit=1, cloud_assist=request.cloud_assist,
                previous_workflow_id=request.previous_workflow_id))
            links[key] = workflow["id"]
            write_import(record)
        workflows.append({"configuration":name, "id":links[key]})
    return {"workflows":workflows}


class GitHubDatasetRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)


@app.post("/datasets/github")
async def github_dataset(request: GitHubDatasetRequest) -> dict:
    from github_import import import_github_dataset
    import urllib.error
    try:
        return await asyncio.to_thread(import_github_dataset, request.url, DATASETS_ROOT)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    except urllib.error.HTTPError as error:
        raise HTTPException(422, "GitHub could not provide this file. Check that the file and branch exist and the repository is public. Redirects and private repositories are not supported.") from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise HTTPException(503, "GitHub is unreachable or the download timed out. Try again.") from error


@app.get("/advisor/status")
async def advisor_status() -> dict:
    return cloud_status()


@app.get("/compass/check")
async def compass_check(model: str = "") -> dict:
    if not compass_client.configured():
        raise HTTPException(503, "Compass is not configured. Set ADVISOR_API_KEY and the Compass ADVISOR_BASE_URL in the trainer's .env, then restart the trainer.")
    if not model.strip() or len(model) > 200:
        raise HTTPException(422, "Choose a Compass model ID of at most 200 characters.")
    error = await asyncio.to_thread(compass_client.check_failure, model)
    if error and error.definite:
        compass_client.drop_available(DATA_ROOT, model)
    return {"ok": error is None, "message": str(error) if error else ""}


# The add list and each run's Compass options come from the saved check; see compass_client.load_available.
compass_availability: dict = {"status": "idle", "models": [], "checked": 0, "total": 0, "checkedAt": None}


def check_compass_catalog() -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    try:
        catalog = compass_client.text_models()
        compass_availability.update(status="checking", checked=0, total=len(catalog), error=None)
        answered = []
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(compass_client.check_failure, model): model for model in catalog}
            for future in as_completed(futures):
                if future.result() is None:
                    answered.append(futures[future])
                compass_availability["checked"] += 1
        if catalog and not answered:
            # Most likely a Compass outage; saving an empty list would hide every model until the next recheck.
            raise compass_client.CompassError("No Compass model answered the check, so the saved list was kept.")
        compass_availability.update(compass_client.save_available(DATA_ROOT, answered), status="complete")
    except Exception as error:
        compass_availability.update(status="failed", error=compass_client.safe_message(error))


@app.get("/compass/available")
async def compass_available(refresh: bool = False) -> dict:
    if not compass_client.configured():
        raise HTTPException(503, "Compass is not configured on the trainer.")
    if compass_availability["status"] != "checking":
        # Reread the file each time: a run that finds a model refused takes it off the saved list.
        saved = compass_client.load_available(DATA_ROOT)
        if saved and not refresh:
            compass_availability.update(saved, status="complete")  # Any error from a failed recheck stays to be shown.
        else:
            compass_availability.update(status="checking", checked=0, total=0)
            asyncio.get_running_loop().run_in_executor(None, check_compass_catalog)
    return compass_availability


@app.post("/advisor/connection-test")
async def advisor_connection_test() -> dict:
    return connection_test()


@app.post("/datasets/analyze")
async def analyze_dataset(request: AnalysisRequest) -> dict:
    try:
        dataset = load_source(request.dataset.source, request.dataset.name, request.dataset.extension)
        analysis = local_profile(dataset)
    except Exception as error:
        # Dataset libraries raise several provider-specific exception classes
        # (for example legacy dataset-script errors). They are invalid inputs,
        # not trainer crashes, so return the actionable message to step 1.
        raise HTTPException(422, str(error)) from error
    cloud = None
    if request.cloud_assist and analysis["classification"]["confidence"] < .8:
        cloud = cloud_clarify(analysis["_samples"], analysis)
        analysis["redaction"]["sentToCloud"] = cloud is not None
        if cloud and cloud.get("task") in {"assistant", "writing", "code", "structured", "classification"}:
            analysis["classification"] = {
                "task": cloud["task"],
                "confidence": min(.9, float(cloud.get("confidence", .6))),
                "source": "cloud-assisted",
                "explanation": cloud.get("explanation", ""),
            }
    candidates = rank_candidates(analysis, request.goal, gpu_status()["memoryGb"])
    blockers = [item for item in analysis["findings"] if item["severity"] == "blocker"]
    recommended = next((item for item in candidates if not item["rejected"]), None)
    return {
        "facts": analysis["facts"],
        "classification": analysis["classification"],
        "findings": analysis["findings"],
        "redaction": analysis["redaction"],
        "cloud": {"requested": request.cloud_assist, "used": cloud is not None, "status": "not checked" if not request.cloud_assist else "used" if cloud else "unavailable_or_abstained"},
        "candidates": candidates,
        "blockers": blockers,
        "recommendation": None if blockers else recommended,
    }


@app.post("/datasets/analyze/baseline")
async def analyze_baseline(request: BaselineRequest) -> dict:
    """Run a small local zero-shot base-model comparison on a fixed representative slice."""
    analysis = await analyze_dataset(AnalysisRequest(dataset=request.dataset, goal=request.goal))
    eligible = [item for item in analysis["candidates"] if not item["rejected"]]
    if request.candidate_ids:
        eligible = [item for item in eligible if item["model"]["id"] in request.candidate_ids]
    shortlisted = eligible[: min(3, len(eligible))]
    if not shortlisted:
        raise HTTPException(422, "No model is feasible for this dataset and GPU.")
    try:
        dataset = load_source(request.dataset.source, request.dataset.name, request.dataset.extension)
        comparisons = baseline_compare(dataset, shortlisted, min(100, max(20, analysis["facts"]["validRows"] // 10)))
    except Exception as error:
        comparisons = [{"status": "unavailable", "message": str(error)[:300]}]
    for index, candidate in enumerate(shortlisted):
        candidate["baseline"] = next((item for item in comparisons if item.get("modelId") == candidate["model"]["id"]), {"status": "unavailable"})
        candidate["baseline"]["rank"] = index + 1
    completed = [item for item in shortlisted if item["baseline"]["status"] == "complete"]
    selected = max(completed, key=lambda item: (item["baseline"].get("meanTokenOverlap") or -1, -item["model"]["sizeB"])) if completed else shortlisted[0]
    return {"shortlist": shortlisted, "finalSelection": selected, "selectionReason": "Same held-out prompt slice compared locally. Token overlap is available only for completion datasets; review output samples before training."}


async def run_training(run_id: str) -> None:
    async with queue_lock:
        record = read_run(run_id)
        if record.get('status') in ('cancelled', 'complete', 'failed'):
            return
        record.update({"status": "running", "startedAt": datetime.now(timezone.utc).isoformat(), "framework": "Transformers + TRL + PEFT"})
        write_run(record)
        command = [sys.executable, "-u", str(ROOT / "train.py"), "--run-dir", str(run_path(run_id).parent)]
        process = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        active_processes[run_id] = process
        log_path = run_path(run_id).parent / "training.log"
        with log_path.open("w", encoding="utf-8") as log:
            while line := await process.stdout.readline():
                log.write(line.decode(errors="replace"))
                log.flush()
        code = await process.wait()
        active_processes.pop(run_id, None)
        record = read_run(run_id)
        if record["status"] == "cancelled":
            return
        record.update({"status": "complete" if code == 0 else "failed", "completedAt": datetime.now(timezone.utc).isoformat()})
        if code == 0:
            metrics_path = run_path(run_id).parent / "metrics.json"
            if metrics_path.exists():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                record["metrics"] = {
                    **metrics,
                    "eval_loss": metrics.get("eval_loss"),
                    "perplexity": round(float(__import__("math").exp(metrics["eval_loss"])), 3) if metrics.get("eval_loss") is not None else None,
                    "classification": metrics.get("classification"),
                }
                record["framework"] = metrics.get("framework", record["framework"])
            record["artifacts"] = {"adapter": "adapter/", "mergedModel": "merged/", "ollamaModelfile": "exports/Modelfile"}
        write_run(record)


@app.post("/runs", status_code=202)
async def create_run(request: RunRequest) -> dict:
    if "14B" in request.base_model and request.parameter_method != "qlora":
        raise HTTPException(422, "14B models require QLoRA on the configured 24 GB GPU.")
    if request.idempotency_key:
        for path in RUNS_ROOT.glob("*/run.json"):
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if existing.get("idempotencyKey") == request.idempotency_key:
                return existing
    run_id = str(uuid.uuid4())
    record = {
        "id": run_id,
        "status": "queued",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "config": request.model_dump(),
        "hardware": gpu_status(),
        "metrics": {},
        "artifacts": {},
        "workflowId": request.workflow_id,
        "candidateId": request.candidate_id,
        "attempt": request.attempt,
        "idempotencyKey": request.idempotency_key,
        "splitManifestId": request.split_manifest_id,
        "splitManifestPath": request.split_manifest_path,
    }
    write_run(record)
    asyncio.create_task(run_training(run_id))
    return record


def workflow_path(workflow_id: str) -> Path:
    return WORKFLOWS_ROOT / workflow_id / "workflow.json"


def read_workflow(workflow_id: str) -> dict:
    path = workflow_path(workflow_id)
    if not path.exists():
        raise HTTPException(404, "Workflow not found")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise HTTPException(500, "Workflow metadata is invalid") from error


def write_workflow(record: dict) -> None:
    path = workflow_path(record["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")


async def submit_workflow_training(spec: dict) -> dict:
    test_dataset = spec.get("testDataset")
    request = RunRequest(
        dataset=DatasetRequest(**spec["dataset"]),
        base_model=spec["baseModel"],
        parameter_method=spec["parameterMethod"],
        hyperparameters=Hyperparameters(**spec["hyperparameters"]),
        test_dataset=DatasetRequest(**test_dataset) if test_dataset else None,
        split_seed=int(spec.get("splitSeed", 42)),
        workflow_id=spec.get("workflowId"),
        candidate_id=spec.get("candidateId"),
        attempt=int(spec.get("attempt", 0)),
        idempotency_key=spec.get("idempotencyKey"),
        split_manifest_id=spec.get("splitManifestId"),
        split_manifest_path=spec.get("splitManifestPath"),
    )
    return await create_run(request)


async def get_workflow_training(run_id: str) -> dict:
    return read_run(run_id)


def workflow_training_dir(run_id: str) -> Path:
    return run_path(run_id).parent


workflow_services = WorkflowServices(
    data_root=DATA_ROOT,
    gpu_memory_gb=lambda: gpu_status()["memoryGb"],
    submit_training=submit_workflow_training,
    get_training=get_workflow_training,
    training_dir=workflow_training_dir,
    gpu_lock=queue_lock,
    isolate_inference=True,
)
workflow_graph = build_workflow_graph(workflow_services, MemorySaver())
workflow_checkpointer_context = None


@app.on_event("startup")
async def initialize_workflow_checkpointer() -> None:
    """Use durable async SQLite persistence when the trainer is fully installed."""
    global workflow_graph, workflow_checkpointer_context
    for path in (DATASETS_ROOT / ".imports").glob("*.json"):
        import_id = uuid.UUID(path.stem)
        record = read_import(import_id)
        if record["status"] == "running":
            (DATASETS_ROOT / (str(import_id) + ".jsonl.partial")).unlink(missing_ok=True)
            for partial in (DATASETS_ROOT / str(import_id)).glob("*.jsonl.partial"):
                partial.unlink()
            record.update(status="interrupted", error="Trainer restarted before this import finished. Start the import again.")
            write_import(record)
    # Nothing drives a run that was mid-step when the trainer stopped; mark it so it can be resumed from its checkpoint.
    for path in WORKFLOWS_ROOT.glob("*/workflow.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if record.get("status") in {"running", "queued"}:
            record.update(status="interrupted", pendingAction=None, error="The trainer restarted before this step finished.")
            write_workflow(record)
            (path.parent / "progress.json").unlink(missing_ok=True)
    if AsyncSqliteSaver is None:
        return
    try:
        workflow_checkpointer_context = AsyncSqliteSaver.from_conn_string(str(WORKFLOW_DB))
        saver = await workflow_checkpointer_context.__aenter__()
        workflow_graph = build_workflow_graph(workflow_services, saver)
    except Exception:
        workflow_checkpointer_context = None


@app.on_event("shutdown")
async def close_workflow_checkpointer() -> None:
    imports = list(active_imports.values())
    for task, stop in imports:
        stop.set()
    if imports:
        await asyncio.gather(*(task for task, stop in imports), return_exceptions=True)
    if workflow_checkpointer_context is not None:
        await workflow_checkpointer_context.__aexit__(None, None, None)


def _interrupt_value(result: dict) -> dict | None:
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    value = getattr(interrupts[0], "value", interrupts[0])
    return value if isinstance(value, dict) else {"type": "human_review", "message": str(value)}


async def execute_workflow(workflow_id: str, resume_response: dict | None = None, from_checkpoint: bool = False) -> None:
    record = read_workflow(workflow_id)
    record["status"] = "running"
    record["pendingAction"] = None
    write_workflow(record)
    config = {"configurable": {"thread_id": workflow_id}}
    try:
        if from_checkpoint:
            # No input: LangGraph reruns the step that never finished, keeping every step before it.
            result = await workflow_graph.ainvoke(None, config=config)
        elif resume_response is None:
            result = await workflow_graph.ainvoke(record["state"], config=config)
        else:
            result = await workflow_graph.ainvoke(Command(resume=resume_response), config=config)
        state_update = {key: value for key, value in result.items() if key != "__interrupt__"}
        state = {**record.get("state", {}), **state_update}
        try:
            snapshot = await workflow_graph.aget_state(config)
            checkpoint_values = getattr(snapshot, "values", None)
            if isinstance(checkpoint_values, dict):
                state = {**state, **checkpoint_values}
        except (AttributeError, TypeError):
            # The result still contains enough state for in-memory/test savers.
            pass
        pending = _interrupt_value(result)
        record["state"] = state
        if pending:
            record["status"] = "waiting"
            record["pendingAction"] = pending
            state["pending_action"] = pending
        else:
            record["status"] = state.get("status", "complete")
            record["pendingAction"] = None
        write_workflow(record)
    except Exception as error:
        record["status"] = "failed"
        record["pendingAction"] = None
        record["error"] = str(error)[:500]
        record.setdefault("state", {}).setdefault("errors", []).append(str(error)[:500])
        write_workflow(record)
    finally:
        active_workflows.pop(workflow_id, None)


@app.post("/workflow/runs", status_code=202)
async def create_workflow(request: WorkflowRequest) -> dict:
    if request.knowledge_documents:
        from quality_probe import validate_documents
        try:
            request.knowledge_documents = validate_documents(request.knowledge_documents)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
    if request.previous_workflow_id is not None:
        read_workflow(str(request.previous_workflow_id))
    workflow_id = str(uuid.uuid4())
    initial_state = {
        "workflow_id": workflow_id,
        "benchmark_selection": True,
        "knowledge_documents": request.knowledge_documents,
        "analysis_limit": request.analysis_limit,
        "task_description": request.task_description,
        "success_metric": request.success_metric,
        "analysis_only": request.analysis_only,
        "auto_analysis": request.auto_analysis,
        "previous_workflow_id": str(request.previous_workflow_id) if request.previous_workflow_id else None,
        "dataset": request.dataset.model_dump(),
        "test_dataset": request.test_dataset.model_dump() if request.test_dataset else None,
        "goal": request.goal,
        "cloud_assist": request.cloud_assist,
        "split_seed": request.split_seed,
        "max_candidates": request.candidate_limit,
        "max_retries": request.max_retries,
        "status": "queued",
        "attempts": {},
        "training_runs": {},
        "evaluation_results": [],
        "errors": [],
    }
    record = {"id": workflow_id, "status": "queued", "createdAt": datetime.now(timezone.utc).isoformat(), "state": initial_state, "pendingAction": None}
    write_workflow(record)
    active_workflows[workflow_id] = asyncio.create_task(execute_workflow(workflow_id))
    return record


def public_workflow(record):
    if record.get("status") == "complete":
        return record
    hidden = {"baseline_results", "evaluation_results", "comparison"}
    return {**{k:v for k,v in record.items() if k not in hidden},
            "state":{k:v for k,v in record.get("state", {}).items() if k not in hidden}}


@app.get("/workflow/runs")
async def list_workflows() -> list[dict]:
    return [public_workflow(json.loads(path.read_text(encoding="utf-8"))) for path in sorted(WORKFLOWS_ROOT.glob("*/workflow.json"), reverse=True)]


@app.get("/workflow/runs/{workflow_id}")
async def get_workflow(workflow_id: str) -> dict:
    record = read_workflow(workflow_id)
    state = record.get("state", {})
    progress = None
    if workflow_id in active_workflows:
        snapshot = await workflow_graph.aget_state({"configurable":{"thread_id":workflow_id}})
        state = {**state, **getattr(snapshot, "values", {})}
        try:
            progress = json.loads((WORKFLOWS_ROOT / workflow_id / "progress.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass  # No step is reporting progress, or it is mid-write.
    training_records = []
    for path in RUNS_ROOT.glob("*/run.json"):
        run = json.loads(path.read_text(encoding="utf-8"))
        if run.get("workflowId") == workflow_id:
            context_path = path.parent / "training-context.json"
            if context_path.exists():
                run["trainingContext"] = json.loads(context_path.read_text(encoding="utf-8"))
            training_records.append(run)
    return public_workflow({**state, **record, "state":state, "trainingRecords":training_records, "progress":progress, "resumable":await workflow_resumable(workflow_id, record)})


def adapter_archive(run_dir: Path) -> Path:
    """Zips the trained LoRA adapter once; the per-epoch checkpoint folders (optimizer state) stay out."""
    archive = run_dir / "exports" / "adapter.zip"
    if not archive.exists():
        archive.parent.mkdir(exist_ok=True)
        partial = archive.with_suffix(".partial")
        with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED) as bundle:
            for path in (run_dir / "adapter").iterdir():
                if path.is_file():
                    bundle.write(path, path.name)
        partial.replace(archive)
    return archive


@app.get("/workflow/runs/{workflow_id}/export")
async def export_workflow_adapter(workflow_id: str) -> FileResponse:
    record = read_workflow(workflow_id)
    comparison = record.get("comparison") or record.get("state", {}).get("comparison") or {}
    model_id = comparison.get("winner") or next((pair["modelId"] for pair in comparison.get("pairs", []) if pair.get("passed")), None)
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in RUNS_ROOT.glob("*/run.json")]
    runs = [run for run in runs if model_id and run.get("workflowId") == workflow_id and run.get("status") == "complete"
            and run.get("config", {}).get("base_model") == model_id and (run_path(run["id"]).parent / "adapter").is_dir()]
    if not runs:
        raise HTTPException(404, "This run has no verified fine-tuned adapter to export.")
    run_dir = run_path(max(runs, key=lambda run: run.get("completedAt") or "")["id"]).parent
    archive = await asyncio.to_thread(adapter_archive, run_dir)
    return FileResponse(archive, media_type="application/zip", filename=model_id.split("/")[-1] + "-fine-tuned-adapter.zip")


async def workflow_resumable(workflow_id: str, record: dict) -> bool:
    if record.get("status") not in {"interrupted", "failed"} or workflow_id in active_workflows:
        return False
    try:
        snapshot = await workflow_graph.aget_state({"configurable": {"thread_id": workflow_id}})
    except Exception:
        return False  # No checkpoint store, e.g. the in-memory fallback after a restart.
    return bool(getattr(snapshot, "next", ()))


@app.post("/workflow/runs/{workflow_id}/continue", status_code=202)
async def continue_workflow(workflow_id: str) -> dict:
    record = read_workflow(workflow_id)
    if workflow_id in active_workflows:
        raise HTTPException(409, "Workflow is already running.")
    if not await workflow_resumable(workflow_id, record):
        raise HTTPException(409, "This run has no unfinished step to resume. Start a new run instead.")
    record["status"] = "queued"
    record.pop("error", None)
    write_workflow(record)
    active_workflows[workflow_id] = asyncio.create_task(execute_workflow(workflow_id, from_checkpoint=True))
    return public_workflow(record)


async def regrade_workflow(record: dict) -> None:
    """Add grades to saved final answers; never generate answers or train again."""
    state = record["state"]
    try:
        manifest = json.loads(Path(state["split_manifest_path"]).read_text(encoding="utf-8"))
        ref = manifest["evalDataset"]
        dataset = await asyncio.to_thread(load_source, ref["source"], ref["name"], ref.get("extension"))
        base, tuned = state["baseline_results"], state["evaluation_results"]
        state["evaluator_model"] = state.get("evaluator_model") or os.environ.get("EVALUATOR_MODEL") or DEFAULT_MODEL
        async with queue_lock:
            results = await asyncio.to_thread(isolated_inference, evaluate_results, dataset, base + tuned,
                task=state.get("profile", {}).get("classification", {}).get("task", "assistant"),
                model_id=state["evaluator_model"], instruction=state.get("selected_instruction", ""),
                knowledge_documents=state.get("knowledge_documents") if state.get("selected_uses_rag") else None)
        state.update(baseline_results=results[:len(base)], evaluation_results=results[len(base):], accuracy_review=accuracy_agent_review(results))
        state["comparison"] = _compare_results(state)
    except Exception:
        record["accuracyError"] = "Automatic evaluation failed. Existing answers are preserved; check backend availability and the cached judge model."
    finally:
        record["status"] = "complete"
        write_workflow(record)
        active_evaluations.pop(record["id"], None)


@app.post("/workflow/runs/{workflow_id}/evaluate", status_code=202)
async def evaluate_workflow(workflow_id: uuid.UUID) -> dict:
    key = str(workflow_id)
    if key in active_evaluations:
        return {"id": key, "status": "evaluating_accuracy"}
    record = read_workflow(key)
    state = record.get("state", {})
    if record.get("status") not in {"complete", "evaluating_accuracy"} or not state.get("evaluation_results") or not state.get("baseline_results"):
        raise HTTPException(409, "A completed workflow with saved base and tuned answers is required.")
    if active_workflows or active_evaluations or queue_lock.locked():
        raise HTTPException(409, "Wait for the current GPU job to finish.")
    # Preserve the original comparison once before adding grades to historical runs.
    backup = workflow_path(key).with_name("before-automatic-evaluation.json")
    if not backup.exists():
        backup.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    record["status"] = "evaluating_accuracy"
    record.pop("accuracyError", None)
    write_workflow(record)
    active_evaluations[key] = asyncio.create_task(regrade_workflow(record))
    return {"id": key, "status": "evaluating_accuracy"}


@app.post("/workflow/runs/{workflow_id}/resume", status_code=202)
async def resume_workflow(workflow_id: str, request: WorkflowResumeRequest) -> dict:
    record = read_workflow(workflow_id)
    if record.get("status") != "waiting":
        raise HTTPException(409, "Workflow is not waiting for human input.")
    if workflow_id in active_workflows:
        raise HTTPException(409, "Workflow is already running.")
    record["status"] = "queued"
    write_workflow(record)
    active_workflows[workflow_id] = asyncio.create_task(execute_workflow(workflow_id, request.response))
    return public_workflow(record)


# Paperclip (https://github.com/paperclipai/paperclip) "http" adapter: an issue assigned to the forge-tune
# agent starts a workflow from the JSON request in its description; the result goes back as a comment.
PAPERCLIP_STATUS = {"complete": "done", "failed": "blocked", "interrupted": "blocked", "waiting": "blocked"}
paperclip_starting: set[str] = set()


class PaperclipWake(BaseModel):
    runId: str
    agentId: str = ""
    context: dict = Field(default_factory=dict)


def paperclip_call(method: str, path: str, run_id: str, body: dict | None = None) -> dict:
    request = urllib.request.Request(os.environ["PAPERCLIP_API_URL"].rstrip("/") + path, method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + os.environ["PAPERCLIP_API_KEY"], "X-Paperclip-Run-Id": run_id, "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read() or b"{}")


def paperclip_issues() -> dict:
    try:
        return json.loads((DATA_ROOT / "paperclip-issues.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_paperclip_issue(issue_id: str, entry: dict) -> None:
    (DATA_ROOT / "paperclip-issues.json").write_text(json.dumps({**paperclip_issues(), issue_id: entry}, indent=2), encoding="utf-8")


def paperclip_workflow_request(issue: dict) -> WorkflowRequest:
    text = issue.get("description") or ""
    match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    return WorkflowRequest.model_validate_json(match.group(1) if match else text)


def paperclip_report(issue_id: str, entry: dict) -> None:
    """Posts a finished (or paused) workflow back once per status."""
    record = read_workflow(entry["workflowId"])
    status = record.get("status")
    if status not in PAPERCLIP_STATUS or entry.get("reported") == status:
        return
    winner = (record.get("state", {}).get("comparison") or {}).get("winner")
    lines = [f"forge-tune workflow `{record['id']}` finished as **{status}**."]
    if winner:
        lines.append(f"Winning model: `{winner}`. Adapter: `GET /workflow/runs/{record['id']}/export` on the trainer.")
    if record.get("error"):
        lines.append("Error: " + record["error"])
    if record.get("pendingAction"):
        lines.append("Waiting for review in forge-tune: " + str(record["pendingAction"].get("message", "")))
    paperclip_call("PATCH", f"/api/issues/{issue_id}", entry["runId"], {"status": PAPERCLIP_STATUS[status], "comment": "\n\n".join(lines)})
    save_paperclip_issue(issue_id, {**entry, "reported": status})


async def handle_paperclip_wake(wake: PaperclipWake) -> None:
    issue_id = wake.context.get("taskId")
    if not issue_id or issue_id in paperclip_starting:
        return
    entry = paperclip_issues().get(issue_id)
    try:
        if entry is None:
            paperclip_starting.add(issue_id)
            issue = await asyncio.to_thread(paperclip_call, "GET", f"/api/issues/{issue_id}", wake.runId)
            try:
                record = await create_workflow(paperclip_workflow_request(issue))
            except (ValueError, HTTPException) as error:
                detail = str(getattr(error, "detail", "") or error)[:1500]
                await asyncio.to_thread(paperclip_call, "PATCH", f"/api/issues/{issue_id}", wake.runId, {"status": "blocked",
                    "comment": "Put a forge-tune workflow request (the `POST /workflow/runs` JSON body) in the issue description.\n\n" + detail})
                return
            entry = {"workflowId": record["id"], "runId": wake.runId, "reported": None}
            save_paperclip_issue(issue_id, entry)
            await asyncio.to_thread(paperclip_call, "PATCH", f"/api/issues/{issue_id}", wake.runId,
                {"status": "in_progress", "comment": f"Started forge-tune workflow `{record['id']}`."})
        else:
            entry = {**entry, "runId": wake.runId}
        # ponytail: a run resumed in the forge-tune UI after a restart or review is reported on the issue's next wake.
        if task := active_workflows.get(entry["workflowId"]):
            await asyncio.wait({task})
        await asyncio.to_thread(paperclip_report, issue_id, entry)
    except Exception as error:
        print(f"Paperclip wake for issue {issue_id} failed: {error}", file=sys.stderr)
    finally:
        paperclip_starting.discard(issue_id)


@app.post("/paperclip/heartbeat", status_code=202)
async def paperclip_heartbeat(wake: PaperclipWake, background: BackgroundTasks, authorization: str = Header(default="")) -> dict:
    secret = os.environ.get("PAPERCLIP_WEBHOOK_SECRET", "")
    if not (secret and os.environ.get("PAPERCLIP_API_URL") and os.environ.get("PAPERCLIP_API_KEY")):
        raise HTTPException(503, "Set PAPERCLIP_WEBHOOK_SECRET, PAPERCLIP_API_URL and PAPERCLIP_API_KEY in trainer/.env.")
    if not hmac.compare_digest(authorization.encode(), ("Bearer " + secret).encode()):
        raise HTTPException(401, "Wrong Paperclip webhook secret.")
    background.add_task(handle_paperclip_wake, wake)
    return {"accepted": True}


@app.get("/runs")
async def list_runs() -> list[dict]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(RUNS_ROOT.glob("*/run.json"), reverse=True)]


@app.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict:
    return read_run(run_id)


@app.get("/runs/{run_id}/logs")
async def get_logs(run_id: str) -> dict:
    log = run_path(run_id).parent / "training.log"
    return {"lines": log.read_text(encoding="utf-8").splitlines()[-200:] if log.exists() else []}


@app.post("/runs/{run_id}/cancel")
async def cancel_run(run_id: str) -> dict:
    record = read_run(run_id)
    process = active_processes.get(run_id)
    if process:
        process.terminate()
    record["status"] = "cancelled"
    write_run(record)
    return record
