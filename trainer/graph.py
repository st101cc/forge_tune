"""LangGraph orchestration for the ForgeTune multi-candidate workflow.

The graph deliberately keeps large datasets and model outputs on disk.  State
contains only references, summaries, and the small amount of information
needed to resume a workflow after a human approval or a process restart.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, TypedDict

from benchmark import INSTRUCTION, fits_local_window, output_budget, select_model, apply_review, compass_results
from compass_client import CompassError, answer_cases, check_failure, configured, drop_available, load_available, text_models
from evaluator import DEFAULT_MODEL, comparable_evidence, evaluate_results
from model_picker import MAX_COMPASS, MAX_LOCAL, built_in_proposal, propose_models
from model_history import backfill_if_empty, recently_unavailable, record_benchmark, record_check_failure, summary_for
from progress import ProgressFile
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from advisor import (
    baseline_compare,
    cloud_assess_blockers,
    cloud_clarify,
    draft_task_brief,
    evaluate_tuned_run,
    evaluation_messages,
    load_source,
    local_profile,
    input_key,
    privacy_scan,
    rank_candidates,
)


class WorkflowState(TypedDict, total=False):
    workflow_id: str
    benchmark_selection: bool
    ai_model_pick: bool
    model_plan: dict[str, Any]
    model_options: dict[str, list[str]]
    compass_models: list[str]
    model_benchmark: dict[str, Any]
    analysis_limit: int
    task_description: str
    success_metric: str
    knowledge_documents: list[dict[str, Any]]
    prompt_uses_rag: bool
    selected_uses_rag: bool
    analysis_only: bool
    auto_analysis: bool
    auto_trial_start: int
    recommendation: dict[str, Any]
    prompt_trials: list[dict[str, Any]]
    prompt_action: str
    prompt_instruction: str
    selected_instruction: str
    previous_workflow_id: str | None
    data_review_action: str
    clarification: dict[str, Any]
    dataset: dict[str, Any]
    test_dataset: dict[str, Any] | None
    goal: Literal["speed", "balanced", "quality"]
    cloud_assist: bool
    task_draft: dict[str, Any]
    split_seed: int
    max_candidates: int
    max_retries: int
    profile: dict[str, Any]
    privacy_review: dict[str, Any]
    accuracy_review: dict[str, Any]
    evaluator_model: str
    split_manifest_id: str
    split_manifest_path: str
    findings: list[dict[str, Any]]
    candidates: list[dict[str, Any]]
    baseline_results: list[dict[str, Any]]
    approved_plan: dict[str, Any]
    candidate_index: int
    attempts: dict[str, int]
    training_runs: dict[str, dict[str, Any]]
    evaluation_results: list[dict[str, Any]]
    comparison: dict[str, Any]
    status: str
    pending_action: dict[str, Any] | None
    errors: list[str]
    candidate_status: str
    retry_reason: str | None
    blocker_assessment: dict[str, Any]
    blocker_resolution_action: str | None


def accuracy_agent_review(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize available correctness evidence without inventing a score."""
    models: dict[str, dict[str, Any]] = {}
    counts = {r.get("modelId"): sum(x.get("modelId") == r.get("modelId") for x in results) for r in results}
    for index, result in enumerate(results):
        evidence = result.get("qualityEvidence") or {}
        model_id = result.get("modelId")
        score = evidence.get("score")
        coverage = evidence.get("coverage")
        examples = evidence.get("evaluatedExamples")
        if result.get("status") != "complete" or not model_id:
            continue
        try:
            measured = coverage == 1 and int(examples or 0) > 0 and type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1
        except (TypeError, ValueError):
            measured = False
        key = str(model_id) if counts[model_id] == 1 else f"{model_id}:{index}"
        models[key] = {
            "status": ("estimated" if evidence.get("estimated") else "measured") if measured else "needs_rubric",
            "metric": evidence.get("metric"),
            "score": float(score) if measured else None,
            "evaluatedExamples": int(examples or 0),
            "reviewExamples": evidence.get("reviewExamples", 0),
        }
    measured = [item for item in models.values() if item["status"] in {"measured", "estimated"}]
    metric = measured[0]["metric"] if measured and all(item["metric"] == measured[0]["metric"] for item in measured) else None
    if measured and len(measured) == len(models) and metric:
        estimated = any(item["status"] == "estimated" for item in measured)
        return {
            "status": "estimated" if estimated else "measured",
            "metric": metric,
            "models": models,
            "message": "Local judge estimates recorded. Review uncertain answers and calibrate against human grades before deployment." if estimated else "Correctness is measured against the recorded targets.",
        }
    if models:
        return {
            "status": "needs_rubric",
            "metric": None,
            "models": models,
            "message": "Automatic rubric grading is incomplete. Check the judge status and review ambiguous examples; missing grades are not accuracy.",
        }
    return {
        "status": "unavailable",
        "metric": None,
        "models": {},
        "message": "No completed model answers were available for accuracy review.",
    }



def diagnose_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    # Compare only complete results for the same model and development examples.
    baseline = {r.get("modelId"): r for r in trials[0]["results"] if r.get("status") == "complete"}
    ranked = []
    scored = []
    for index, trial in enumerate(trials):
        for result in trial["results"]:
            base = baseline.get(result.get("modelId"))
            if not base or result.get("status") != "complete" or trial.get("developmentIds") != trials[0].get("developmentIds"):
                continue
            overlap = result.get("meanTokenOverlap")
            base_overlap = base.get("meanTokenOverlap")
            fmt, base_fmt = result.get("formatValidRate"), base.get("formatValidRate")
            if base_fmt is not None and (fmt is None or fmt < base_fmt):
                continue
            quality, base_quality = _quality_value(result), _quality_value(base)
            if quality is not None and base_quality is not None and comparable_evidence(result, base):
                scored.append((quality - base_quality, index, result))
            if overlap is not None and base_overlap is not None:
                ranked.append((overlap - base_overlap, index, result))
    ranked = scored or ranked
    best = max(ranked, key=lambda item: (item[0], -item[1])) if ranked else None
    selected = best[1] if best else 0
    improved = bool(best and best[0] >= .05 and selected != 0)
    # Token overlap is a diagnostic, never a reliable measure of factual correctness.
    format_failures = sum(1 for sample in (best[2].get("samples", []) if best else []) if not str(sample.get("baseOutput", "")).strip())
    structured_failure = bool(best and best[2].get("formatValidRate") is not None and (1 - best[2]["formatValidRate"]) * len(best[2].get("samples", [])) >= 3)
    return {
        "title": "Try the improved prompt first" if improved and scored else "Review the automatically tested prompt" if improved else "Consider a small fine-tune pilot" if structured_failure or format_failures >= 3 else "More evidence needed before choosing RAG or fine-tuning",
        "reason": "An automatic prompt increased the recorded correctness score without reducing measured format validity. Model-judged scores are estimates; validate on unseen examples before adopting it." if improved and scored else "The selected prompt increased word overlap, but factual correctness is unmeasured. Review its answers against a task rubric before choosing it." if improved else "Repeated output-format or empty-answer failures remain after automatic prompt trials. A small fine-tune is a candidate, not a proven solution." if structured_failure or format_failures >= 3 else "These trials do not establish a reliable behavioral improvement or prove that knowledge is missing. Token overlap alone cannot distinguish factual errors from valid alternative answers.",
        "rag": "Tested against the connected references. Inspect retrieved sources and actual answers; retrieval coverage and word overlap do not prove correctness." if any(t.get("usesRag") for t in trials) else "Not tested: no independent knowledge source is connected. Supply relevant documents to test retrieval; reference answers must not be used as retrieved context for their own evaluation questions.",
        "selectedTrial": selected,
        "confidence": "provisional",
    }

@dataclass
class WorkflowServices:
    """Side effects supplied by the FastAPI application.

    Keeping these callbacks outside the graph makes routing unit-testable and
    prevents the graph from importing the FastAPI module back into itself.
    """

    data_root: Path
    gpu_memory_gb: Callable[[], int]
    submit_training: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
    get_training: Callable[[str], Awaitable[dict[str, Any]]]
    training_dir: Callable[[str], Path]
    gpu_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    isolate_inference: bool = False


def _start_inference_worker() -> None:
    import multiprocessing
    import os
    import threading
    import truststore
    # Spawned workers start without the API's certificate store patch, and may download models.
    truststore.inject_into_ssl()
    # A killed trainer leaves its worker blocked on the task queue, holding GPU memory; exit as soon as the parent is gone.
    parent = multiprocessing.parent_process()
    if parent is not None:
        threading.Thread(target=lambda: (parent.join(), os._exit(1)), daemon=True).start()


def isolated_inference(function, *args, **kwargs):
    # A process boundary releases CUDA contexts as well as tensors after evaluation.
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"), initializer=_start_inference_worker) as pool:
        return pool.submit(function, *args, **kwargs).result()


def _dataset_key(dataset: dict[str, Any]) -> str:
    return json.dumps(
        {
            "source": dataset.get("source"),
            "name": dataset.get("name"),
            "extension": dataset.get("extension"),
            "split": dataset.get("split", "train"),
        },
        sort_keys=True,
    )


def _content(row: dict[str, Any]) -> str:
    if isinstance(row.get("messages"), list):
        return "\n".join(str(item.get("content", "")) if isinstance(item, dict) else str(item) for item in row["messages"])
    if "prompt" in row or "completion" in row:
        return str(row.get("prompt", "")) + "\n" + str(row.get("completion", ""))
    return str(row.get("text", ""))


def _valid_indices(dataset: Any, schema: str | None) -> list[int]:
    if not schema:
        return []
    valid: list[int] = []
    for index, row in enumerate(dataset):
        if _content(row).strip():
            valid.append(index)
    return valid


def _blocker_assessment(state: WorkflowState, cloud_advice: dict[str, Any] | None = None) -> dict[str, Any]:
    """Conservative remediation policy. The agent may advise, but never bypasses it."""
    blockers = [item for item in state.get("findings", []) if item.get("severity") == "blocker"]
    codes = {item.get("code") for item in blockers}
    upload = state.get("dataset", {}).get("source") == "upload"
    cleanup_codes = {"invalid_rows", "duplicates"}
    can_materialize = bool(codes) and codes.issubset(cleanup_codes) and upload
    if can_materialize:
        decision = "approval_required"
        operations = (["drop_invalid_rows"] if "invalid_rows" in codes else []) + (["deduplicate_exact_rows"] if "duplicates" in codes else [])
        reason = "The uploaded dataset can be copied to a cleaned derivative, but this removes rows and requires approval."
    else:
        decision = "human_required"
        operations = []
        reason = "This blocker requires a replacement dataset or data collection; no safe automatic remediation is available."
    if cloud_advice and isinstance(cloud_advice.get("reason"), str):
        reason += " Cloud advisor: " + cloud_advice["reason"][:300]
    return {
        "decision": decision,
        "requiresHumanApproval": True,
        "canMaterializeAfterApproval": can_materialize,
        "operations": operations,
        "blockers": blockers,
        "reason": reason,
        "advisor": {"used": cloud_advice is not None, "suggestedOperations": cloud_advice.get("suggestedOperations", []) if cloud_advice else []},
    }


def _clean_uploaded_dataset(dataset: Any, profile: dict[str, Any], destination: Path) -> dict[str, Any]:
    """Create an auditable JSONL derivative after the user approves cleanup."""
    schema = profile.get("facts", {}).get("schema")
    kept, seen = [], set()
    for index in _valid_indices(dataset, schema):
        row = dataset[index]
        digest = hashlib.sha256(" ".join(_content(row).lower().split()).encode()).hexdigest()
        if digest not in seen:
            kept.append(index)
            seen.add(digest)
    if len(kept) < 30:
        raise ValueError("Approved cleanup leaves fewer than 30 valid rows; replace or expand the dataset instead.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset.select(kept).to_json(str(destination))
    return {"keptRows": len(kept), "removedRows": len(dataset) - len(kept), "path": str(destination)}


def _advisory(default: Any, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    # The model history only advises; a failed read or write must never fail a workflow step.
    try:
        return function(*args, **kwargs)
    except Exception:
        return default


def _persist_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _make_split_manifest(state: WorkflowState, services: WorkflowServices, dataset: Any, profile: dict[str, Any]) -> dict[str, Any]:
    workflow_dir = services.data_root / "workflows" / state["workflow_id"]
    seed = int(state.get("split_seed", 42))
    schema = profile.get("facts", {}).get("schema")
    train_indices = _valid_indices(dataset, schema)
    test_ref = state.get("test_dataset")
    # ponytail: exact normalized-input groups; semantic near-duplicate detection needs a separate audit.
    groups = {}
    for index in train_indices:
        groups.setdefault(input_key(dataset[index]), []).append(index)
    shuffled = list(groups.values())
    random.Random(seed).shuffle(shuffled)

    if test_ref:
        eval_dataset = load_source(test_ref["source"], test_ref["name"], test_ref.get("extension"))
        eval_indices = _valid_indices(eval_dataset, schema)
        if any(input_key(eval_dataset[i]) in groups for i in eval_indices):
            raise ValueError("Training and external test inputs overlap. Supply an independent test dataset.")
        train_ref = state["dataset"]
        eval_ref = test_ref
    else:
        eval_count = max(1, round(len(shuffled) * 0.1)) if shuffled else 0
        eval_indices = sorted(i for group in shuffled[:eval_count] for i in group)
        shuffled = shuffled[eval_count:]
        train_ref = state["dataset"]
        eval_ref = state["dataset"]

    dev_count = max(1, len(shuffled) // 10) if shuffled else 0
    dev_indices = [i for group in shuffled[:dev_count] for i in group]
    train_indices = sorted(i for group in shuffled[dev_count:] for i in group)
    evaluation_ids = random.Random(seed + 2).sample(eval_indices, min(len(eval_indices), state.get("analysis_limit", 50)))
    payload = {
        "developmentIndices": dev_indices,
        "workflowId": state["workflow_id"],
        "seed": seed,
        "schema": schema,
        "datasetFingerprint": hashlib.sha256(_dataset_key(state["dataset"]).encode()).hexdigest(),
        "trainDataset": train_ref,
        "evalDataset": eval_ref,
        "trainIndices": train_indices,
        "evalIndices": eval_indices,
        "evaluationIds": evaluation_ids,
        "leakageCheck": {"method": "normalized exact input grouping", "inputGroups": len(groups), "duplicateInputRows": sum(len(g) - 1 for g in groups.values())},
        "warnings": (["Fewer than 30 development or final examples: exploratory results, not reliable model-selection evidence."] if min(len(dev_indices), len(evaluation_ids)) < 30 else []),
    }
    path = workflow_dir / "split.json"
    _persist_json(path, payload)
    payload["path"] = str(path)
    return payload


def _plan_error(models: Any, local: list[str], compass: list[str]) -> str | None:
    if not isinstance(models, list) or not all(isinstance(m, dict) for m in models):
        return "Choose the models to benchmark."
    for m in models:
        if m.get("provider") not in ("local", "compass") or m.get("id") not in (local if m["provider"] == "local" else compass):
            return f"{str(m.get('id'))[:200]} is not an available {m.get('provider')} model."
    if len({m["id"] for m in models}) != len(models):
        return "Approve each model only once."
    if not 2 <= len(models) <= MAX_LOCAL + MAX_COMPASS:
        return f"Approve between 2 and {MAX_LOCAL + MAX_COMPASS} models to compare."
    hosted = sum(m["provider"] == "compass" for m in models)
    if len(models) - hosted > MAX_LOCAL or hosted > MAX_COMPASS:
        return f"Approve at most {MAX_LOCAL} local and {MAX_COMPASS} Compass models."
    return None


def _candidate_key(candidate: dict[str, Any]) -> str:
    return str(candidate["model"]["id"])


def _current_candidate(state: WorkflowState) -> dict[str, Any]:
    candidates = state.get("approved_plan", {}).get("candidates") or state.get("candidates", [])
    return candidates[int(state.get("candidate_index", 0))]


def _default_hyperparameters(profile: dict[str, Any]) -> dict[str, Any]:
    valid_rows = int(profile.get("facts", {}).get("validRows", 0))
    p95 = int(profile.get("facts", {}).get("estimatedTokenLength", {}).get("p95", 1024))
    return {
        "rank": 8 if valid_rows < 500 else 16,
        "alpha": 16 if valid_rows < 500 else 32,
        "learning_rate": 2e-4,
        "epochs": 2 if valid_rows < 500 else 1 if valid_rows > 50000 else 3,
        "batch_size": 1,
        "gradient_accumulation": 8,
        "max_sequence_length": min(4096, max(1024, p95 * 2)),
    }


def _base_plan(state: WorkflowState) -> dict[str, Any]:
    plan_candidates = []
    for item in state.get("candidates", [])[: int(state.get("max_candidates", 3))]:
        hyperparameters = _default_hyperparameters(state["profile"])
        # ponytail: 7B activations exceed a 15 GB T4 at long context; raise the cap when larger GPUs are supported.
        if int(item.get("model", {}).get("sizeB", 0) or 0) >= 7:
            hyperparameters["max_sequence_length"] = min(hyperparameters["max_sequence_length"], 1024)
        plan_candidates.append(
            {
                "model": item["model"],
                "method": item["method"],
                "hyperparameters": hyperparameters,
                "estimatedVramGb": item.get("estimatedVramGb"),
                "estimatedDuration": item.get("estimatedDuration"),
                "score": item.get("score"),
                "reasons": item.get("reasons", []),
            }
        )
    return {
        "candidates": plan_candidates,
        "candidateCount": len(plan_candidates),
        "splitManifestId": state.get("split_manifest_id"),
        "goal": state.get("goal", "balanced"),
        "baselineResults": state.get("baseline_results", []),
        "maxRetries": int(state.get("max_retries", 2)),
    }


def _validate_plan(plan: dict[str, Any], memory_gb: int) -> tuple[bool, str | None]:
    candidates = plan.get("candidates")
    if not isinstance(candidates, list) or not candidates or len(candidates) > 3:
        return False, "Choose between one and three candidates."
    seen: set[str] = set()
    for candidate in candidates:
        model = candidate.get("model", {})
        model_id = model.get("id")
        if not model_id or model_id in seen:
            return False, "Each candidate must have a unique model ID."
        seen.add(model_id)
        if float(candidate.get("estimatedVramGb", model.get("vram", 0)) or 0) > memory_gb:
            return False, f"{model_id} exceeds the available GPU memory."
        if "14B" in model_id and candidate.get("method") != "qlora":
            return False, "14B models require QLoRA on the configured GPU."
        if candidate.get("method") not in {"lora", "qlora"}:
            return False, "Training method must be LoRA or QLoRA."
    return True, None


def _metric_value(result: dict[str, Any], name: str, default: float | None = None) -> float | None:
    value = result.get(name)
    try:
        return float(value) if value is not None and math.isfinite(float(value)) else default
    except (TypeError, ValueError):
        return default


def _quality_value(result: dict[str, Any]) -> float | None:
    evidence = result.get("qualityEvidence", {})
    if evidence.get("coverage") != 1 or not evidence.get("evaluatedExamples"):
        return None
    ids = result.get("evaluationIds", [])
    if ids and evidence["evaluatedExamples"] != len(ids):
        return None
    value = _metric_value(evidence, "score")
    return value if value is not None and 0 <= value <= 1 else None


def _compare_results(state: WorkflowState) -> dict[str, Any]:
    baselines = {item.get("modelId"): item for item in state.get("baseline_results", [])}
    tuned = {item.get("modelId"): item for item in state.get("evaluation_results", [])}
    pairs: list[dict[str, Any]] = []
    for candidate in state.get("approved_plan", {}).get("candidates", []):
        model_id = candidate["model"]["id"]
        base = baselines.get(model_id, {"status": "unavailable"})
        result = tuned.get(model_id, {"status": "unavailable"})
        base_overlap = _metric_value(base, "meanTokenOverlap")
        tuned_overlap = _metric_value(result, "meanTokenOverlap")
        base_format = _metric_value(base, "formatValidRate")
        tuned_format = _metric_value(result, "formatValidRate")
        base_loss = _metric_value(base, "eval_loss")
        tuned_loss = _metric_value(result, "eval_loss")
        base_latency = _metric_value(base, "latencySeconds")
        tuned_latency = _metric_value(result, "latencySeconds")
        base_quality, tuned_quality = _quality_value(base), _quality_value(result)
        same_metric = comparable_evidence(base, result)
        same_examples = bool(base.get("evaluationIds")) and base.get("evaluationIds") == result.get("evaluationIds")
        quality_delta = tuned_quality - base_quality if same_metric and same_examples and base_quality is not None and tuned_quality is not None else None
        overlap_delta = tuned_overlap - base_overlap if tuned_overlap is not None and base_overlap is not None else None
        pairs.append(
            {
                "modelId": model_id,
                "base": base,
                "tuned": result,
                "qualityScoreDelta": round(quality_delta, 4) if quality_delta is not None else None,
                "qualityMetric": base.get("qualityEvidence", {}).get("metric") if quality_delta is not None else None,
                "meanTokenOverlapDelta": round(overlap_delta, 4) if overlap_delta is not None else None,
                "formatValidRateDelta": round(tuned_format - base_format, 4) if tuned_format is not None and base_format is not None else None,
                "evalLossDelta": round(tuned_loss - base_loss, 4) if tuned_loss is not None and base_loss is not None else None,
                "latencyDeltaSeconds": round(tuned_latency - base_latency, 3) if tuned_latency is not None and base_latency is not None else None,
                "passed": result.get("status") == "complete" and result.get("artifactValid", False),
            }
        )

    goal = state.get("goal", "balanced")
    # Correctness scores must cover the same held-out examples; overlap is diagnostic only.
    comparable = [item for item in pairs if item["passed"] and item["base"].get("status") == "complete" and item["qualityScoreDelta"] is not None]
    valid = [item for item in comparable if item["qualityScoreDelta"] >= 0.05 and (item["formatValidRateDelta"] is None or item["formatValidRateDelta"] >= 0)]

    def rank_key(item: dict[str, Any]) -> tuple[float, float, float]:
        tuned_result = item["tuned"]
        quality = _quality_value(tuned_result)
        quality = quality if quality is not None else -1.0
        latency = _metric_value(tuned_result, "latencySeconds", 1e9) or 1e9
        vram = _metric_value(tuned_result, "estimatedVramGb", 1e9) or 1e9
        if goal == "speed":
            return (latency, -quality, vram)
        if goal == "quality":
            return (-quality, vram, latency)
        return (-quality + (latency / 1000), latency, vram)

    leaderboard = sorted(valid, key=rank_key)
    manifest = json.loads(Path(state["split_manifest_path"]).read_text(encoding="utf-8"))
    return {
        "goal": goal,
        "pairs": pairs,
        "leaderboard": leaderboard,
        "winner": leaderboard[0]["modelId"] if leaderboard else None,
        "decision": "review_fine_tune" if leaderboard else "keep_baseline" if comparable else "review_required",
        "decisionReason": "At least 5 percentage points of correctness-score improvement on the same held-out examples, using the same evaluator with full grading coverage and no measured format regression, are required for review. Local model grades are estimates requiring human calibration. This is not deployment approval.",
        "evaluationIds": manifest.get("evaluationIds", manifest.get("evalIndices", [])[:100]),
        "accuracyReview": state.get("accuracy_review"),
        "privacyReview": state.get("privacy_review"),
        "completedAt": time.time(),
    }


def build_workflow_graph(services: WorkflowServices, checkpointer: Any) -> Any:
    """Build and compile the resumable ForgeTune graph."""

    async def run_inference(function, *args, **kwargs):
        if services.isolate_inference:
            return await asyncio.to_thread(isolated_inference, function, *args, **kwargs)
        return await asyncio.to_thread(function, *args, **kwargs)

    # The graph saves state only when a node returns, so long steps write this file for the page's live progress.
    def progress_path(state):
        return services.data_root / "workflows" / state["workflow_id"] / "progress.json"

    def report_progress(state, message):
        try:
            _persist_json(progress_path(state), {"message": message, "startedAt": time.time()})
        except OSError:
            pass  # Progress is best effort; it must never fail a workflow step.

    def clear_progress(state):
        progress_path(state).unlink(missing_ok=True)

    def counted_progress(state, message, base, total):
        # Fills the page's bar from base to base + the phase's questions, out of total for the whole step.
        return ProgressFile(progress_path(state), message, base, total)

    async def grade_results(state, dataset, results, instruction="", knowledge_documents=None, on_progress=None):
        async with services.gpu_lock:
            return await run_inference(evaluate_results, dataset, results,
                task=state.get("profile", {}).get("classification", {}).get("task", "assistant"),
                model_id=state.get("evaluator_model", DEFAULT_MODEL), instruction=instruction,
                knowledge_documents=knowledge_documents, on_progress=on_progress)

    async def initialize(state: WorkflowState) -> dict[str, Any]:
        return {
            "status": "profiling",
            "candidate_index": 0,
            "attempts": state.get("attempts", {}),
            "training_runs": state.get("training_runs", {}),
            "evaluation_results": state.get("evaluation_results", []),
            "errors": state.get("errors", []),
            "max_candidates": min(3, int(state.get("max_candidates", 3))),
            "max_retries": min(2, int(state.get("max_retries", 2))),
            "split_seed": int(state.get("split_seed", 42)),
            "evaluator_model": state.get("evaluator_model") or os.environ.get("EVALUATOR_MODEL") or DEFAULT_MODEL,
        }

    async def load_and_normalize(state: WorkflowState) -> dict[str, Any]:
        dataset_ref = state["dataset"]
        dataset = await asyncio.to_thread(load_source, dataset_ref["source"], dataset_ref["name"], dataset_ref.get("extension"))
        columns = list(dataset.column_names)
        sample = [_content(dataset[index])[:1000] for index in range(min(100, len(dataset)))]
        fingerprint = hashlib.sha256((_dataset_key(dataset_ref) + json.dumps(sample, ensure_ascii=False)).encode()).hexdigest()
        workflow_dir = services.data_root / "workflows" / state["workflow_id"]
        _persist_json(workflow_dir / "dataset.json", {"reference": dataset_ref, "columns": columns, "rows": len(dataset), "fingerprint": fingerprint})
        return {"status": "profiling", "dataset": {**dataset_ref, "rows": len(dataset), "columns": columns, "fingerprint": fingerprint}}

    async def profile_and_split(state: WorkflowState) -> dict[str, Any]:
        dataset_ref = state["dataset"]
        dataset = await asyncio.to_thread(load_source, dataset_ref["source"], dataset_ref["name"], dataset_ref.get("extension"))
        profile = await asyncio.to_thread(local_profile, dataset)
        manifest = await asyncio.to_thread(_make_split_manifest, state, services, dataset, profile)
        return {
            "profile": profile,
            "findings": profile.get("findings", []),
            "split_manifest_id": hashlib.sha256(str(manifest["path"]).encode()).hexdigest()[:16],
            "split_manifest_path": manifest["path"],
            "status": "planning",
        }

    async def privacy_scanner(state: WorkflowState) -> dict[str, Any]:
        dataset_ref = state["dataset"]
        dataset = await asyncio.to_thread(load_source, dataset_ref["source"], dataset_ref["name"], dataset_ref.get("extension"))
        indices = sorted(random.Random(42).sample(range(len(dataset)), min(600, len(dataset))))
        review = privacy_scan(list(dataset.select(indices)))
        review["flaggedRows"] = [indices[i] for i in review["flaggedRows"]]
        return {"privacy_review": review}

    def route_after_profile(state: WorkflowState) -> str:
        blockers = [item for item in state.get("findings", []) if item.get("severity") == "blocker"]
        confidence = float(state.get("profile", {}).get("classification", {}).get("confidence", 0))
        if blockers:
            return "assess_blockers"
        uncertain = confidence < 0.8 or any(item.get("code") == "uncertain_task" for item in state.get("findings", []))
        if uncertain and state.get("cloud_assist"):
            return "clarify_data"
        if uncertain:
            return "human_data_review"
        return "dataset_ready" if state.get("analysis_only") else "build_candidates"

    async def assess_blockers(state: WorkflowState) -> dict[str, Any]:
        blockers = [item for item in state.get("findings", []) if item.get("severity") == "blocker"]
        advice = cloud_assess_blockers(state.get("profile", {}).get("_samples", []), state.get("profile", {}), blockers) if state.get("cloud_assist") else None
        assessment = _blocker_assessment(state, advice)
        return {"blocker_assessment": assessment, "status": "awaiting_blocker_review"}

    async def blocker_resolution_review(state: WorkflowState) -> dict[str, Any]:
        assessment = state.get("blocker_assessment", {})
        actions = ["replace_dataset", "abort"]
        if assessment.get("canMaterializeAfterApproval"):
            actions.insert(0, "approve_remediation")
        response = interrupt(
            {
                "type": "blocker_resolution",
                "message": assessment.get("reason", "A dataset blocker needs review."),
                "assessment": assessment,
                "actions": actions,
            }
        )
        if not isinstance(response, dict):
            raise ValueError("Blocker resolution response must be an object.")
        action = response.get("action")
        if action == "abort":
            return {"status": "blocked", "pending_action": None, "errors": ["Workflow aborted during blocker resolution."]}
        if action == "replace_dataset":
            dataset = response.get("dataset")
            if not isinstance(dataset, dict) or not dataset.get("source") or not dataset.get("name"):
                raise ValueError("replace_dataset requires a dataset source and name.")
            return {"dataset": dataset, "pending_action": None, "status": "profiling", "blocker_resolution_action": "replace_dataset"}
        if action == "approve_remediation" and assessment.get("canMaterializeAfterApproval"):
            return {"pending_action": None, "status": "remediating", "blocker_resolution_action": "approve_remediation"}
        raise ValueError("Unsupported blocker resolution action.")

    async def apply_blocker_remediation(state: WorkflowState) -> dict[str, Any]:
        if state.get("dataset", {}).get("source") != "upload":
            raise ValueError("Only uploaded datasets can be remediated in place.")
        dataset_ref = state["dataset"]
        dataset = load_source(dataset_ref["source"], dataset_ref["name"], dataset_ref.get("extension"))
        destination = services.data_root / "workflows" / state["workflow_id"] / "remediated.jsonl"
        result = _clean_uploaded_dataset(dataset, state["profile"], destination)
        return {
            "dataset": {"source": "upload", "name": result["path"], "extension": "jsonl", "displayName": "remediated.jsonl"},
            "status": "profiling",
            "errors": state.get("errors", []) + [f"Approved remediation removed {result['removedRows']} rows and kept {result['keptRows']}."],
            "blocker_resolution_action": "remediated",
        }

    def route_after_blocker_review(state: WorkflowState) -> str:
        if state.get("status") == "blocked":
            return "finish_blocked"
        if state.get("blocker_resolution_action") == "replace_dataset":
            return "load_and_normalize"
        return "apply_blocker_remediation"

    async def human_data_review(state: WorkflowState) -> dict[str, Any]:
        validation_error = None
        while True:
            response = interrupt({
                "type": "data_review",
                "message": validation_error or "The intended task is ambiguous. What should a model learn from these examples?",
                "profile": state.get("profile", {}), "findings": state.get("findings", []),
                "actions": ["replace_dataset", "accept_classification", "abort"],
            })
            if not isinstance(response, dict):
                validation_error = "Choose a task and describe the intended result."
                continue
            action = response.get("action")
            if action == "abort":
                return {"status": "blocked", "pending_action": None}
            if action == "replace_dataset":
                dataset = response.get("dataset")
                if not isinstance(dataset, dict) or not dataset.get("source") or not dataset.get("name"):
                    validation_error = "A replacement dataset source and name are required."
                    continue
                return {"dataset": dataset, "clarification": {}, "pending_action": None, "status": "profiling", "data_review_action": "replace_dataset"}
            task, notes = response.get("task"), response.get("notes", "")
            if action != "accept_classification" or not isinstance(task, str) or task not in {"assistant", "writing", "code", "structured", "classification"} or not isinstance(notes, str) or not notes.strip() or len(notes) > 2000:
                validation_error = "Select a task and provide a short explanation (1–2000 characters)."
                continue
            if any(item.get("severity") == "blocker" for item in state.get("findings", [])):
                raise ValueError("Dataset blockers must be resolved first.")
            profile = {**state["profile"], "classification": {"task": task, "confidence": 1.0, "source": "user-confirmed", "explanation": notes.strip()}}
            return {"profile": profile, "clarification": {"task": task, "notes": notes.strip()}, "pending_action": None, "status": "planning", "data_review_action": "accept_classification"}

    def route_after_data_review(state: WorkflowState) -> str:
        if state.get("status") == "blocked":
            return "finish_blocked"
        if state.get("data_review_action") == "replace_dataset":
            return "load_and_normalize"
        return "dataset_ready" if state.get("analysis_only") else "build_candidates"

    async def clarify_data(state: WorkflowState) -> dict[str, Any]:
        profile = state["profile"]
        cloud = cloud_clarify(profile.get("_samples", [])[:20], profile)
        if not cloud or cloud.get("task") not in {"assistant", "writing", "code", "structured", "classification"}:
            return {
                "status": "planning",
                "errors": ["Cloud clarification was unavailable or abstained."],
                "pending_action": None,
                "cloud_clarification_failed": True,
            }
        updated = dict(profile)
        updated["classification"] = {
            "task": cloud["task"],
            "confidence": min(0.9, float(cloud.get("confidence", 0.6))),
            "source": "cloud-assisted",
            "explanation": cloud.get("explanation", ""),
        }
        updated["redaction"] = {**updated.get("redaction", {}), "sentToCloud": True}
        return {"profile": updated, "status": "planning"}

    def route_after_clarification(state: WorkflowState) -> str:
        # An agent suggestion does not replace user confirmation of ambiguous intent.
        return "human_data_review"

    async def dataset_ready(state: WorkflowState) -> dict[str, Any]:
        if state.get("auto_analysis"):
            return {"analysis_only": False, "status": "planning"}
        while True:
            response = interrupt({"type": "dataset_ready", "message": "Dataset analysis is complete. Review the findings before starting model evaluation.", "profile": state.get("profile", {}), "actions": ["continue_training", "abort"]})
            if isinstance(response, dict) and response.get("action") == "abort":
                return {"status": "cancelled", "pending_action": None}
            if isinstance(response, dict) and response.get("action") == "continue_training":
                return {"analysis_only": False, "auto_analysis": True, "status": "planning", "pending_action": None}

    async def draft_task(state: WorkflowState) -> dict[str, Any]:
        # Runs once before the task form: LangGraph reruns an interrupted node on every resume, so the cloud is asked only here.
        if "task_description" not in state or state.get("task_draft") or all(str(state.get(key) or "").strip() for key in ("task_description", "success_metric")):
            return {}
        cloud = bool(state.get("cloud_assist"))
        if cloud:
            report_progress(state, "Drafting the task description")
        try:
            draft = await asyncio.to_thread(draft_task_brief, state.get("profile", {}), cloud)
        finally:
            if cloud:
                clear_progress(state)
        return {"task_draft": draft}

    async def build_candidates(state: WorkflowState) -> dict[str, Any]:
        # Persisted workflows predating task briefs retain their existing decisions.
        brief = {key: state.get(key, "") for key in ("task_description", "success_metric")}
        limit = state.get("analysis_limit", 50)
        pick = state.get("ai_model_pick", True)
        while "task_description" in state and not all(isinstance(v, str) and 1 <= len(v.strip()) <= 2000 for v in brief.values()):
            response = interrupt({"type": "task_review", "message": "Define the intended task and how success will be measured before comparing models.", "analysisLimit": limit, "draft": state.get("task_draft")})
            if not isinstance(response, dict):
                continue
            if response.get("action") == "abort":
                return {"status": "cancelled", "candidates": [], "pending_action": None}
            brief = {key: response.get(key, "") for key in brief}
            chosen = response.get("analysis_limit", limit)
            pick = response.get("ai_model_pick", True)
            if type(chosen) is not int or not 1 <= chosen <= 200 or type(pick) is not bool:
                brief = {key: "" for key in brief}
                continue
            limit = chosen
        if "task_description" in state:
            manifest = json.loads(Path(state["split_manifest_path"]).read_text(encoding="utf-8"))
            manifest["evaluationIds"] = random.Random(state.get("split_seed", 42) + 2).sample(manifest["evalIndices"], min(len(manifest["evalIndices"]), limit))
            _persist_json(Path(state["split_manifest_path"]), manifest)
        memory = services.gpu_memory_gb()
        ranked = rank_candidates(state["profile"], state.get("goal", "balanced"), memory)
        eligible = [item for item in ranked if not item.get("rejected")]
        if not eligible:
            return {"candidates": [], "status": "blocked", "errors": ["No model is feasible for the available GPU memory."]}
        return {**({key: value.strip() for key, value in brief.items()} if "task_description" in state else {}), "analysis_limit": limit, "ai_model_pick": pick, "candidates": eligible if state.get("benchmark_selection") else eligible[: int(state.get("max_candidates", 3))], "status": "benchmarking" if state.get("benchmark_selection") else "baseline"}

    async def pick_models(state: WorkflowState) -> dict[str, Any]:
        # Runs once before the pause: LangGraph re-runs an interrupted node on every resume, so Compass is asked only here.
        # The catalog (GET /models) carries no brief or data, so it loads whenever Compass is configured; the tick box gates only the agent.
        picking = state.get("ai_model_pick", True)
        memory = services.gpu_memory_gb()
        # The full ranking, so models that do not fit are shown as excluded; the proposal helpers keep only eligible ones.
        ranked = rank_candidates(state["profile"], state.get("goal", "balanced"), memory)
        compass, proposal, skipped = [], None, {}
        note = "AI picking is off for this run." if not picking else "Compass is not configured on the trainer."
        task = state["profile"]["classification"]["task"]
        await asyncio.to_thread(_advisory, None, backfill_if_empty, services.data_root)
        if configured():
            report_progress(state, "Asking Compass which models to compare")
            try:
                # Models that definitely failed to run in the last 7 days are hidden from the agent and the add list.
                catalog = await asyncio.to_thread(text_models)
                # Once the whole catalog has been checked, offer only the models that answered, so nothing needs checking per run.
                saved = load_available(services.data_root)
                if saved is not None:
                    catalog = [model for model in catalog if model in saved["models"]]
                recent = _advisory({}, recently_unavailable, services.data_root)
                skipped = {model: recent[model] for model in catalog if model in recent}
                compass = [model for model in catalog if model not in skipped]
                if picking:
                    brief = {key: state.get(key) for key in ("task_description", "success_metric", "analysis_limit")}
                    proposal = await asyncio.to_thread(propose_models, brief, state.get("profile", {}), memory, ranked, compass, history=_advisory([], summary_for, services.data_root, task))
                unchecked = [m for m in proposal["models"] if m["provider"] == "compass" and not (saved and m["id"] in saved["models"])] if proposal and proposal["source"] == "agent" else []
                if unchecked:
                    # Without a saved check the catalog may list models this key cannot run, so check the proposed ones.
                    report_progress(state, "Checking that the proposed Compass models respond")
                    left_out = 0
                    for model in unchecked:
                        failure = await asyncio.to_thread(check_failure, model["id"])
                        if failure:
                            left_out += 1
                            proposal["models"].remove(model)
                            # Only a definite refusal bans the model for this run; a temporary failure can be added back by hand.
                            if failure.definite:
                                proposal["excluded"].append({"id": model["id"], "reason": f"Compass could not run it: {failure}"})
                                compass.remove(model["id"])
                                _advisory(None, record_check_failure, services.data_root, task, model["id"], str(failure), state["workflow_id"])
                            else:
                                proposal["excluded"].append({"id": model["id"], "reason": f"Compass did not respond in time: {str(failure).rstrip('.')}. You can add it back below."})
                    if left_out:
                        proposal["note"] = " ".join(filter(None, [proposal["note"], f"{left_out} proposed Compass model(s) were left out because they failed a quick check; see Not suggested."]))
            except CompassError as error:
                note = f"Compass models are unavailable: {error}; showing the built-in local ranking."
            finally:
                clear_progress(state)
        plan = proposal or built_in_proposal(ranked, note)
        plan["excluded"] += [{"id": model, "reason": f"Couldn't run on {past['date']:%d %b}: {past['error'] or 'Compass refused it'} — skipped for 7 days."} for model, past in skipped.items()]
        return {"model_plan": plan, "model_options": {"local": [c["model"]["id"] for c in state["candidates"]], "compass": compass}}

    async def approve_models(state: WorkflowState) -> dict[str, Any]:
        local, compass = state["model_options"]["local"], state["model_options"]["compass"]
        error = None
        while True:
            response = interrupt({"type": "model_plan", "message": error or "Approve or change the models to benchmark.", "proposal": state["model_plan"], "localOptions": local, "compassOptions": compass, "privacyFlagged": (state.get("privacy_review") or {}).get("status") == "review"})
            if isinstance(response, dict) and response.get("action") == "abort":
                return {"status": "cancelled", "pending_action": None}
            models = response.get("models") if isinstance(response, dict) and response.get("action") == "approve" else None
            error = _plan_error(models, local, compass)
            if error:
                continue
            approved = {m["id"] for m in models if m["provider"] == "local"}
            return {"candidates": [c for c in state["candidates"] if c["model"]["id"] in approved], "compass_models": [m["id"] for m in models if m["provider"] == "compass"], "status": "benchmarking", "pending_action": None}

    async def benchmark_models(state: WorkflowState) -> dict[str, Any]:
        ref=state["dataset"]
        dataset=load_source(ref["source"],ref["name"],ref.get("extension"))
        manifest=json.loads(Path(state["split_manifest_path"]).read_text(encoding="utf-8"))
        development=manifest.get("developmentIndices",[])
        fitting=[i for i in development if fits_local_window(dataset[i])]
        ids=fitting[:state.get("analysis_limit",50)]
        results=[]
        if not ids:
            return {"status":"blocked","errors":["No development questions are available for model selection." if not development else "Every development question is too long for the local models' 4096-token window."]}
        max_tokens=output_budget([evaluation_messages(dataset[i],INSTRUCTION)[1] for i in ids])
        hosted=state.get("compass_models",[])
        total=len(state["candidates"])+len(hosted)
        # Each model answers then gets graded on every question: half the bar answering, half grading.
        units=2*total*len(ids)
        try:
            async with services.gpu_lock:
                for index,candidate in enumerate(state["candidates"],1):
                    progress=counted_progress(state,f"Answering with {candidate['model']['id']} · model {index} of {total} · {len(ids)} questions",(index-1)*len(ids),units)
                    try:
                        records=await run_inference(baseline_compare,dataset,[candidate],len(ids),eval_indices=ids,task=state["profile"]["classification"]["task"],instruction=INSTRUCTION,max_new_tokens=max_tokens,strict_context=True,on_progress=progress)
                        record=records[0]
                    except Exception as error:
                        record={"status":"unavailable","message":str(error)[:300]}
                    results.append({**record,"modelId":candidate["model"]["id"],"provider":"local"})
            # Hosted answers need no GPU; build cases from the same development questions so compass_results accepts them.
            for index,model in enumerate(hosted,len(state["candidates"])+1):
                progress=counted_progress(state,f"Answering with {model} on Compass · model {index} of {total} · {len(ids)} questions",(index-1)*len(ids),units)
                try:
                    rows=[dataset[i] for i in ids]
                    cases=[{"rowId":i,"messages":[m for m in evaluation_messages(row,INSTRUCTION)[0] if not (m["role"]=="system" and not m["content"].strip())]} for i,row in zip(ids,rows)]
                    answers=await asyncio.to_thread(answer_cases,model,cases,on_answer=progress,max_tokens=max_tokens)
                    job={"phase":"development","status":"complete","instruction":INSTRUCTION,"maxOutputTokens":max_tokens,"splitManifestId":state["split_manifest_id"],"models":[model],"cases":cases,"results":answers}
                    results+=compass_results(job,{"developmentIds":ids,"splitManifestId":state["split_manifest_id"],"maxOutputTokens":max_tokens},rows)
                except Exception as error:  # One failing hosted model must not discard the local answers, as for local models above.
                    row={"status":"unavailable","message":str(error)[:300],"modelId":model,"provider":"compass","unavailableDefinite":False}
                    # The request carried dataset text, so a 4xx may be about this dataset; only a refused dataset-free check bans the model.
                    if getattr(error,"definite",False):
                        check=await asyncio.to_thread(check_failure,model)
                        if check:
                            row.update(unavailableDefinite=check.definite,checkMessage=str(check)[:300])
                            if check.definite:
                                _advisory(None,drop_available,services.data_root,model)
                    results.append(row)
            # Failed models have nothing to grade, so grading spans only the answers that came back.
            grading=total*len(ids)
            graded=sum(len(r.get("evaluationIds",[])) for r in results if r.get("status")=="complete")
            progress=counted_progress(state,f"Grading the answers with {state.get('evaluator_model') or DEFAULT_MODEL}",grading,grading+graded)
            results = await grade_results(state, dataset, results, INSTRUCTION, on_progress=progress)
        finally:
            clear_progress(state)
        results,prices=apply_review(results,{},{})
        summary=select_model(results,ids)
        summary["accuracyReview"] = accuracy_agent_review(results)
        _advisory(None,record_benchmark,services.data_root,state["profile"]["classification"]["task"],results,state["workflow_id"])
        return {"model_benchmark":{**summary,"splitManifestId":state["split_manifest_id"],"instruction":INSTRUCTION,"maxOutputTokens":max_tokens,"tooLongQuestions":len(development)-len(fitting),"prices":prices},"status":"awaiting_model_selection"}

    async def model_selection(state: WorkflowState) -> dict[str, Any]:
        benchmark=state["model_benchmark"]
        winner=benchmark.get("winner")
        if winner:
            selected=next(r for r in benchmark["results"] if r["modelId"]==winner)
            if selected["provider"]=="compass":
                return {"status":"complete","comparison":{"decision":"keep_baseline","pairs":[],"modelIds":[winner],"instruction":INSTRUCTION,"decisionReason":"The development benchmark selected a hosted Compass model. This model is retained as the baseline; local fine-tuning is not applicable. Validate it on untouched final questions before deployment."}}
            return {"candidates":[c for c in state["candidates"] if c["model"]["id"]==winner],"status":"baseline"}
        response=interrupt({"type":"model_selection","message":benchmark.get("error") or benchmark["reason"],"benchmark":benchmark,"actions":["select_best","abort"]})
        if not isinstance(response,dict):
            return {"model_benchmark":{**benchmark,"error":"Review the model benchmark."}}
        if response.get("action")=="abort": return {"status":"cancelled"}
        try:
            results=benchmark["results"]
            if response.get("action")!="select_best": raise ValueError("Choose a benchmark action.")
            if len(results)>MAX_LOCAL+MAX_COMPASS: raise ValueError(f"This pilot compares at most {MAX_LOCAL+MAX_COMPASS} models.")
            results,prices=apply_review(results,response,benchmark.get("prices",{}))
            summary=select_model(results,benchmark["developmentIds"])
            summary["accuracyReview"] = accuracy_agent_review(results)
            return {"model_benchmark":{**benchmark,**summary,"prices":prices,"error":None},"status":"awaiting_model_selection"}
        except (ValueError,TypeError,KeyError) as error:
            return {"model_benchmark":{**benchmark,"error":str(error)[:400]}}

    async def baseline_evaluate(state: WorkflowState) -> dict[str, Any]:
        dataset_ref = state["dataset"]
        dataset = load_source(dataset_ref["source"], dataset_ref["name"], dataset_ref.get("extension"))
        manifest = json.loads(Path(state["split_manifest_path"]).read_text(encoding="utf-8"))
        candidates = state.get("candidates", [])[: int(state.get("max_candidates", 3))]
        indices = manifest.get("developmentIndices", manifest.get("trainIndices", [])[:10])[:state.get("analysis_limit", 10)]
        instruction = state.get("prompt_instruction", "")
        label = state.get("progress_label") or ("Prompt test" if instruction else "Baseline test")
        models = ", ".join(c["model"]["id"] for c in candidates)
        # Automatic analysis runs several tests in a row; each fills its share of one bar.
        part, parts = state.get("progress_part", (1, 1))
        answers = len(candidates) * len(indices)
        start, units = (part - 1) * 2 * answers, parts * 2 * answers
        try:
            progress = counted_progress(state, f"{label}: answering with {models} · {len(indices)} questions", start, units)
            try:
                async with services.gpu_lock:
                    comparisons = await run_inference(baseline_compare, dataset, candidates, len(indices), eval_indices=indices, task=state.get("profile", {}).get("classification", {}).get("task"), instruction=instruction, knowledge_documents=state.get("knowledge_documents") if state.get("prompt_uses_rag") else None, on_progress=progress)
            except Exception as error:
                comparisons = [{"status": "unavailable", "message": str(error)[:300]}]
            progress = counted_progress(state, f"{label}: grading the answers with {state.get('evaluator_model') or DEFAULT_MODEL}", start + answers, units)
            comparisons = await grade_results(state, dataset, comparisons, instruction, state.get("knowledge_documents") if state.get("prompt_uses_rag") else None, on_progress=progress)
        finally:
            clear_progress(state)
        trials = state.get("prompt_trials", []) + [{"instruction": instruction, "usesRag": bool(state.get("prompt_uses_rag")), "knowledgeDocuments": state.get("knowledge_documents", []) if state.get("prompt_uses_rag") else [], "results": comparisons, "developmentIds": indices}]
        return {"baseline_results": comparisons, "prompt_trials": trials, "accuracy_review": accuracy_agent_review(comparisons), "status": "awaiting_prompt_review"}

    async def accuracy_reviewer(state: WorkflowState) -> dict[str, Any]:
        return {"accuracy_review": accuracy_agent_review(state.get("baseline_results", []))}

    async def automatic_analysis(state: WorkflowState) -> dict[str, Any]:
        trials = list(state.get("prompt_trials", []))
        completed = [r for r in trials[-1]["results"] if r.get("status") == "complete"] if trials else []
        if not completed:
            return {"recommendation": {"title": "Model evaluation unavailable", "reason": "The local inference runtime could not run the baseline. No prompt, RAG, or fine-tuning conclusion is supported yet. Retry automatic analysis after connecting a working runtime.", "rag": "Not tested: no independent knowledge source is connected.", "selectedTrial": None}, "status": "awaiting_prompt_review"}
        task = state.get("profile", {}).get("classification", {}).get("task", "instruction")
        rules = {
            "structured": "Return only valid JSON. Do not add Markdown fences or commentary. Follow the requested schema exactly.",
            "classification": "Return only the requested label. Use only the labels allowed by the task.",
            "code": "Return correct code for the requested language and constraints. Handle the stated edge cases. Do not invent APIs.",
            "writing": "Follow the requested tone, length, and structure. Preserve the supplied facts.",
            "summarization": "Summarize the supplied material faithfully. Preserve essential facts and do not introduce unsupported claims.",
        }
        instruction = rules.get(task, "Follow the user's task and requested output format exactly. Give a direct, concise answer and do not invent missing facts.")
        # ponytail: two local prompt strategies; replace with a learned optimizer when reliable task-specific scoring exists.
        prompts = [instruction, instruction + " Check your answer for missing requirements and contradictions before returning the final answer. If the supplied information is insufficient, state what is missing."]
        total = len(prompts) + bool(state.get("knowledge_documents"))
        for number, prompt in enumerate(prompts, 1):
            result = await baseline_evaluate({**state, "prompt_trials": trials, "prompt_instruction": prompt, "prompt_uses_rag": False, "progress_label": f"Prompt test {number} of {total}", "progress_part": (number, total)})
            trials = result["prompt_trials"]
        if state.get("knowledge_documents"):
            result = await baseline_evaluate({**state, "prompt_trials":trials, "prompt_instruction":instruction, "prompt_uses_rag":True, "progress_label": f"Prompt test {total} of {total} (with references)", "progress_part": (total, total)})
            trials = result["prompt_trials"]
        start = state.get("auto_trial_start", 0)
        recommendation = diagnose_trials(trials[start:])
        recommendation["selectedTrial"] += start
        latest_results = trials[-1].get("results", []) if trials else []
        return {"prompt_trials": trials, "accuracy_review": accuracy_agent_review(latest_results), "recommendation": recommendation, "status": "awaiting_prompt_review"}

    async def prompt_review(state: WorkflowState) -> dict[str, Any]:
        error = None
        trials = state.get("prompt_trials", [])
        while True:
            response = interrupt({"type":"prompt_review", "message":error or "Compare prompts on examples prepared from this dataset. Choose the best configuration before deciding whether to fine-tune.", "trials":trials, "recommendation":state.get("recommendation"), "actions":["auto_analyze", "test_rag", "try_prompt", "keep_baseline", "fine_tune", "abort"]})
            if not isinstance(response, dict):
                error = "Choose a prompt-review action."
                continue
            action = response.get("action")
            if action == "abort":
                return {"status":"cancelled", "prompt_action":"abort"}
            if action == "auto_analyze":
                return {"auto_analysis":True, "prompt_uses_rag":False, "prompt_instruction":"", "auto_trial_start":len(trials), "prompt_action":"auto_analyze", "status":"baseline", "recommendation":{}}
            if action == "test_rag":
                from quality_probe import validate_documents
                try:
                    documents = validate_documents(response.get("documents"))
                except ValueError as invalid:
                    error = str(invalid)
                    continue
                return {"knowledge_documents":documents, "prompt_uses_rag":False, "auto_analysis":True, "prompt_uses_rag":False, "prompt_instruction":"", "auto_trial_start":len(trials), "prompt_action":"auto_analyze", "status":"baseline"}
            if action == "try_prompt":
                instruction, context = response.get("instruction", ""), response.get("context", "")
                if not isinstance(instruction, str) or not isinstance(context, str) or len(instruction) > 4000 or len(context) > 8000:
                    error = "Use instructions up to 4000 characters and context up to 8000 characters."
                    continue
                if len(trials) >= 10:
                    error = "This workflow has reached its 10 prompt trials. Choose a result or start a new dataset version."
                    continue
                combined = instruction + ("\n\nReference context:\n" + context if context.strip() else "")
                return {"auto_analysis":False, "recommendation":{}, "prompt_instruction":combined, "prompt_uses_rag":False, "prompt_action":"try_prompt", "status":"baseline"}
            index = response.get("trial_index", state.get("recommendation", {}).get("selectedTrial"))
            if action not in {"keep_baseline", "fine_tune"} or not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(trials):
                error = "Select a recorded prompt version to review."
                continue
            selected = trials[index]
            if selected.get("usesRag") and "knowledgeDocuments" not in selected:
                error = "This older RAG trial has no saved references. Test the references again before selecting it."
                continue
            results = [item for item in selected["results"] if item.get("status") == "complete"]
            if not results:
                error = "This trial has no completed inference. Retry when the local model runtime is available."
                continue
            update = {"selected_instruction":selected["instruction"], "selected_uses_rag":bool(selected.get("usesRag")), "baseline_results":results, "prompt_action":action, "pending_action":None}
            if selected.get("usesRag"):
                update["knowledge_documents"] = selected["knowledgeDocuments"]
            if action == "keep_baseline":
                update.update({"status":"complete", "comparison":{"decision":"keep_baseline", "selectedTrial":index, "instruction":selected["instruction"], "modelIds":[item["modelId"] for item in results], "pairs":[], "decisionReason":"You reviewed the dataset examples and chose to keep this untuned configuration."}})
            else:
                update.update({"status":"awaiting_plan", "candidates":[item for item in state.get("candidates", []) if any(result.get("modelId") == item["model"]["id"] for result in results)]})
            return update

    async def select_and_approve_plan(state: WorkflowState) -> dict[str, Any]:
        proposed = _base_plan(state)
        validation_error = None
        while True:
            response = interrupt(
                {
                    "type": "plan_approval",
                    "message": validation_error or "Approve or edit the model and training plan before GPU work begins.",
                    "plan": proposed,
                    "actions": ["approve", "edit", "abort"],
                    "validationError": validation_error,
                }
            )
            if not isinstance(response, dict):
                validation_error = "Plan approval response must be an object."
                continue
            if response.get("action") == "abort":
                return {"status": "cancelled", "pending_action": None}
            plan = response.get("plan") if response.get("action") == "edit" else proposed
            valid, reason = _validate_plan(plan, services.gpu_memory_gb())
            if not valid:
                validation_error = reason or "Invalid training plan."
                proposed = plan if isinstance(plan, dict) else proposed
                continue
            return {"approved_plan": plan, "candidate_index": 0, "status": "training", "pending_action": None}

    def route_after_plan(state: WorkflowState) -> str:
        return "finish_cancelled" if state.get("status") == "cancelled" else "prepare_candidate"

    async def prepare_candidate(state: WorkflowState) -> dict[str, Any]:
        candidate = _current_candidate(state)
        key = _candidate_key(candidate)
        attempts = dict(state.get("attempts", {}))
        attempts.setdefault(key, 0)
        return {"attempts": attempts, "candidate_status": "prepared", "retry_reason": None}

    async def submit_training(state: WorkflowState) -> dict[str, Any]:
        candidate = _current_candidate(state)
        key = _candidate_key(candidate)
        attempt = int(state.get("attempts", {}).get(key, 0))
        job_key = f"{state['workflow_id']}:{key}:{attempt}"
        existing = state.get("training_runs", {}).get(job_key)
        if existing and existing.get("runId"):
            return {"candidate_status": "submitted"}
        hp = dict(candidate.get("hyperparameters") or _default_hyperparameters(state["profile"]))
        record = await services.submit_training(
            {
                "workflowId": state["workflow_id"],
                "candidateId": key,
                "attempt": attempt,
                "idempotencyKey": job_key,
                "dataset": state["dataset"],
                "testDataset": state.get("test_dataset"),
                "baseModel": key,
                "parameterMethod": candidate["method"],
                "hyperparameters": hp,
                "splitManifestId": state["split_manifest_id"],
                "splitManifestPath": state["split_manifest_path"],
                "splitSeed": state.get("split_seed", 42),
            }
        )
        training_runs = dict(state.get("training_runs", {}))
        training_runs[job_key] = {"runId": record["id"], "candidateId": key, "attempt": attempt, "status": record.get("status", "queued")}
        return {"training_runs": training_runs, "candidate_status": "submitted"}

    async def wait_for_training(state: WorkflowState) -> dict[str, Any]:
        candidate = _current_candidate(state)
        key = _candidate_key(candidate)
        attempt = int(state.get("attempts", {}).get(key, 0))
        job_key = f"{state['workflow_id']}:{key}:{attempt}"
        run_id = state.get("training_runs", {}).get(job_key, {}).get("runId")
        if not run_id:
            return {"candidate_status": "failed", "retry_reason": "Training job was not recorded."}
        started = time.monotonic()
        while True:
            record = await services.get_training(run_id)
            status = record.get("status")
            if status in {"complete", "failed", "cancelled"}:
                training_runs = dict(state.get("training_runs", {}))
                training_runs[job_key] = {**training_runs.get(job_key, {}), "status": status}
                return {"training_runs": training_runs, "candidate_status": "complete" if status == "complete" else "failed", "retry_reason": None if status == "complete" else f"Training ended with status {status}."}
            if time.monotonic() - started > 24 * 60 * 60:
                return {"candidate_status": "failed", "retry_reason": "Training exceeded the 24-hour workflow timeout."}
            await asyncio.sleep(2)

    def route_after_training(state: WorkflowState) -> str:
        return "verify_artifacts" if state.get("candidate_status") == "complete" else "assess_retry"

    async def verify_artifacts(state: WorkflowState) -> dict[str, Any]:
        candidate = _current_candidate(state)
        key = _candidate_key(candidate)
        attempt = int(state.get("attempts", {}).get(key, 0))
        job_key = f"{state['workflow_id']}:{key}:{attempt}"
        run_id = state.get("training_runs", {}).get(job_key, {}).get("runId")
        record = await services.get_training(run_id) if run_id else {}
        run_dir = services.training_dir(run_id) if run_id else Path()
        metrics_path = run_dir / "metrics.json"
        adapter_dir = run_dir / "adapter"
        valid = bool(record.get("status") == "complete" and metrics_path.exists() and adapter_dir.exists())
        if valid:
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                valid = all(value is None or math.isfinite(float(value)) for value in metrics.values() if isinstance(value, (int, float)))
            except (OSError, ValueError, TypeError):
                valid = False
        return {"candidate_status": "verified" if valid else "failed", "retry_reason": None if valid else "Training artifacts are missing or invalid."}

    def route_after_verification(state: WorkflowState) -> str:
        return "evaluate_tuned" if state.get("candidate_status") == "verified" else "assess_retry"

    async def evaluate_tuned(state: WorkflowState) -> dict[str, Any]:
        candidate = _current_candidate(state)
        key = _candidate_key(candidate)
        attempt = int(state.get("attempts", {}).get(key, 0))
        job_key = f"{state['workflow_id']}:{key}:{attempt}"
        run_id = state.get("training_runs", {}).get(job_key, {}).get("runId")
        record = await services.get_training(run_id) if run_id else {}
        metrics_path = services.training_dir(run_id) / "metrics.json" if run_id else Path()
        metrics: dict[str, Any] = {}
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        baseline = {"modelId": key, "status": "unavailable"}
        manifest = json.loads(Path(state["split_manifest_path"]).read_text(encoding="utf-8"))
        eval_indices = manifest.get("evaluationIds", manifest.get("evalIndices", [])[:100])
        try:
            eval_dataset = None
            if state.get("test_dataset"):
                test_ref = state["test_dataset"]
                eval_dataset = load_source(test_ref["source"], test_ref["name"], test_ref.get("extension"))
            final_dataset = eval_dataset if eval_dataset is not None else load_source(state["dataset"]["source"], state["dataset"]["name"], state["dataset"].get("extension"))
            async with services.gpu_lock:
                final_baselines = await run_inference(baseline_compare, final_dataset, [candidate], len(eval_indices), eval_indices=eval_indices, task=state.get("profile", {}).get("classification", {}).get("task"), instruction=state.get("selected_instruction", ""), knowledge_documents=state.get("knowledge_documents") if state.get("selected_uses_rag") else None)
                baseline = final_baselines[0]
                generated = await run_inference(evaluate_tuned_run, final_dataset, candidate, services.training_dir(run_id), eval_indices, task=state.get("profile", {}).get("classification", {}).get("task"), instruction=state.get("selected_instruction", ""), knowledge_documents=state.get("knowledge_documents") if state.get("selected_uses_rag") else None)
        except Exception as error:
            generated = {"status": "unavailable", "message": str(error)[:300], "modelId": key}
        tuned_overlap = _metric_value(generated, "meanTokenOverlap")
        base_overlap = _metric_value(baseline, "meanTokenOverlap")
        tuned_format = _metric_value(generated, "formatValidRate")
        base_format = _metric_value(baseline, "formatValidRate")
        tuned_loss = _metric_value(generated, "eval_loss") or _metric_value(metrics, "eval_loss")
        base_loss = _metric_value(baseline, "eval_loss")
        tuned_quality, base_quality = _quality_value(generated), _quality_value(baseline)
        quality_regression = bool(
            (tuned_quality is not None and base_quality is not None and tuned_quality < base_quality)
            or (tuned_format is not None and base_format is not None and tuned_format < base_format)
        )
        eval_loss = tuned_loss
        evaluation = {
            **generated,
            "modelId": key,
            "runId": run_id,
            "artifactValid": generated.get("status") == "complete",
            "eval_loss": eval_loss,
            "perplexity": round(math.exp(eval_loss), 3) if eval_loss is not None and eval_loss < 700 else None,
            "estimatedVramGb": candidate.get("estimatedVramGb"),
            "baseStatus": baseline.get("status", "unavailable"),
            "qualityRegression": quality_regression,
            "trainingRecord": {"status": record.get("status"), "framework": record.get("framework")},
        }
        results = [item for item in state.get("evaluation_results", []) if item.get("modelId") != key]
        results.append(evaluation)
        return {
            "evaluation_results": results,
            "baseline_results": [item for item in state.get("baseline_results", []) if item.get("modelId") != key] + [baseline],
            "candidate_status": "evaluated" if generated.get("status") == "complete" and not quality_regression else "failed",
            "retry_reason": "Measured task correctness or format validity regressed against the base model." if quality_regression else None if generated.get("status") == "complete" else generated.get("message", "Tuned evaluation was unavailable."),
        }

    async def tuned_accuracy_reviewer(state: WorkflowState) -> dict[str, Any]:
        candidate = _current_candidate(state)
        model_id = _candidate_key(candidate)
        baseline = next((item for item in reversed(state.get("baseline_results", [])) if item.get("modelId") == model_id), None)
        tuned = next((item for item in reversed(state.get("evaluation_results", [])) if item.get("modelId") == model_id), None)
        manifest = json.loads(Path(state["split_manifest_path"]).read_text(encoding="utf-8"))
        ref = manifest["evalDataset"]
        dataset = load_source(ref["source"], ref["name"], ref.get("extension"))
        results = await grade_results(state, dataset, [item for item in (baseline, tuned) if item],
            state.get("selected_instruction", ""), state.get("knowledge_documents") if state.get("selected_uses_rag") else None)
        if len(results) != 2:
            return {"accuracy_review": accuracy_agent_review(results)}
        return {"baseline_results": [r for r in state.get("baseline_results", []) if r.get("modelId") != model_id] + [results[0]],
                "evaluation_results": [r for r in state.get("evaluation_results", []) if r.get("modelId") != model_id] + [results[1]],
                "accuracy_review": accuracy_agent_review(results)}

    async def assess_retry(state: WorkflowState) -> dict[str, Any]:
        candidate = _current_candidate(state)
        key = _candidate_key(candidate)
        attempt = int(state.get("attempts", {}).get(key, 0))
        reason = state.get("retry_reason") or "Candidate evaluation failed."
        if attempt < int(state.get("max_retries", 2)):
            attempts = dict(state.get("attempts", {}))
            attempts[key] = attempt + 1
            updated = dict(candidate)
            hp = dict(updated.get("hyperparameters") or _default_hyperparameters(state["profile"]))
            hp["learning_rate"] = max(1e-6, float(hp.get("learning_rate", 2e-4)) / 2)
            hp["epochs"] = max(1, int(hp.get("epochs", 1)) - 1)
            updated["hyperparameters"] = hp
            candidates = list(state.get("approved_plan", {}).get("candidates", []))
            candidates[int(state.get("candidate_index", 0))] = updated
            plan = {**state["approved_plan"], "candidates": candidates}
            return {"attempts": attempts, "approved_plan": plan, "candidate_status": "retry", "retry_reason": reason}
        response = interrupt(
            {
                "type": "retry_review",
                "message": f"Candidate {key} did not produce a valid result after the retry budget.",
                "reason": reason,
                "modelId": key,
                "actions": ["retry", "skip_candidate", "abort"],
            }
        )
        action = response.get("action") if isinstance(response, dict) else None
        if action == "retry":
            attempts = dict(state.get("attempts", {}))
            attempts[key] = attempt + 1
            return {"attempts": attempts, "candidate_status": "retry", "retry_reason": reason}
        if action == "skip_candidate":
            return {"candidate_status": "skipped", "retry_reason": None}
        return {"status": "failed", "candidate_status": "failed", "errors": [reason]}

    def route_after_retry(state: WorkflowState) -> str:
        if state.get("status") == "failed":
            return "finish_failed"
        return "submit_training" if state.get("candidate_status") == "retry" else "advance_candidate"

    async def advance_candidate(state: WorkflowState) -> dict[str, Any]:
        next_index = int(state.get("candidate_index", 0)) + 1
        total = len(state.get("approved_plan", {}).get("candidates", []))
        return {"candidate_index": next_index, "status": "training" if next_index < total else "comparing"}

    def route_after_advance(state: WorkflowState) -> str:
        return "prepare_candidate" if int(state.get("candidate_index", 0)) < len(state.get("approved_plan", {}).get("candidates", [])) else "compare_final"

    async def compare_final(state: WorkflowState) -> dict[str, Any]:
        return {"comparison": _compare_results(state), "status": "complete", "pending_action": None}

    async def finish(state: WorkflowState) -> dict[str, Any]:
        return {"pending_action": None}

    builder = StateGraph(WorkflowState)
    nodes = {
        "initialize": initialize,
        "dataset_ready": dataset_ready,
        "load_and_normalize": load_and_normalize,
        "profile_and_split": profile_and_split,
        "privacy_scanner": privacy_scanner,
        "assess_blockers": assess_blockers,
        "blocker_resolution_review": blocker_resolution_review,
        "apply_blocker_remediation": apply_blocker_remediation,
        "human_data_review": human_data_review,
        "clarify_data": clarify_data,
        "build_candidates": build_candidates,
        "pick_models": pick_models,
        "approve_models": approve_models,
        "benchmark_models": benchmark_models,
        "model_selection": model_selection,
        "baseline_evaluate": baseline_evaluate,
        "accuracy_reviewer": accuracy_reviewer,
        "prompt_review": prompt_review,
        "automatic_analysis": automatic_analysis,
        "select_and_approve_plan": select_and_approve_plan,
        "prepare_candidate": prepare_candidate,
        "submit_training": submit_training,
        "wait_for_training": wait_for_training,
        "verify_artifacts": verify_artifacts,
        "evaluate_tuned": evaluate_tuned,
        "tuned_accuracy_reviewer": tuned_accuracy_reviewer,
        "assess_retry": assess_retry,
        "advance_candidate": advance_candidate,
        "compare_final": compare_final,
        "draft_task": draft_task,
        "finish_blocked": finish,
        "finish_cancelled": finish,
        "finish_failed": finish,
    }
    for name, node in nodes.items():
        builder.add_node(name, node)

    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "load_and_normalize")
    builder.add_edge("load_and_normalize", "profile_and_split")
    builder.add_edge("profile_and_split", "privacy_scanner")
    builder.add_conditional_edges("privacy_scanner", route_after_profile, {"assess_blockers": "assess_blockers", "human_data_review": "human_data_review", "clarify_data": "clarify_data", "build_candidates": "draft_task", "dataset_ready": "dataset_ready"})
    builder.add_edge("assess_blockers", "blocker_resolution_review")
    builder.add_conditional_edges("blocker_resolution_review", route_after_blocker_review, {"load_and_normalize": "load_and_normalize", "apply_blocker_remediation": "apply_blocker_remediation", "finish_blocked": "finish_blocked"})
    builder.add_edge("apply_blocker_remediation", "load_and_normalize")
    builder.add_conditional_edges("human_data_review", route_after_data_review, {"load_and_normalize": "load_and_normalize", "build_candidates": "draft_task", "dataset_ready": "dataset_ready", "finish_blocked": "finish_blocked"})
    builder.add_conditional_edges("clarify_data", route_after_clarification, {"human_data_review": "human_data_review", "build_candidates": "draft_task"})
    builder.add_conditional_edges("build_candidates", lambda state: "finish_blocked" if not state.get("candidates") else "pick_models" if state.get("benchmark_selection") else "baseline_evaluate", {"finish_blocked": "finish_blocked", "pick_models": "pick_models", "baseline_evaluate": "baseline_evaluate"})
    builder.add_edge("pick_models", "approve_models")
    builder.add_conditional_edges("approve_models", lambda state: "finish_cancelled" if state.get("status") == "cancelled" else "benchmark_models", {"finish_cancelled": "finish_cancelled", "benchmark_models": "benchmark_models"})
    builder.add_conditional_edges("dataset_ready", lambda state: "finish_cancelled" if state.get("status") == "cancelled" else "build_candidates", {"finish_cancelled": "finish_cancelled", "build_candidates": "draft_task"})
    builder.add_edge("draft_task", "build_candidates")
    builder.add_conditional_edges("benchmark_models",lambda state: "finish_blocked" if state.get("status")=="blocked" else "model_selection")
    builder.add_conditional_edges("model_selection",lambda state: END if state.get("status") in {"complete","cancelled"} else "baseline_evaluate" if state.get("status")=="baseline" else "model_selection")
    builder.add_edge("baseline_evaluate", "accuracy_reviewer")
    builder.add_conditional_edges("accuracy_reviewer", lambda state: "automatic_analysis" if state.get("auto_analysis") else "prompt_review")
    builder.add_edge("automatic_analysis", "prompt_review")
    builder.add_conditional_edges("prompt_review", lambda state: state.get("prompt_action"), {"auto_analyze":"baseline_evaluate", "try_prompt":"baseline_evaluate", "fine_tune":"select_and_approve_plan", "keep_baseline":END, "abort":"finish_cancelled"})
    builder.add_conditional_edges("select_and_approve_plan", route_after_plan, {"prepare_candidate": "prepare_candidate", "finish_cancelled": "finish_cancelled"})
    builder.add_edge("prepare_candidate", "submit_training")
    builder.add_edge("submit_training", "wait_for_training")
    builder.add_conditional_edges("wait_for_training", route_after_training, {"verify_artifacts": "verify_artifacts", "assess_retry": "assess_retry"})
    builder.add_conditional_edges("verify_artifacts", route_after_verification, {"evaluate_tuned": "evaluate_tuned", "assess_retry": "assess_retry"})
    builder.add_edge("evaluate_tuned", "tuned_accuracy_reviewer")
    # Final outcomes are report-only: never use held-out quality or judge failures to retrain.
    builder.add_edge("tuned_accuracy_reviewer", "advance_candidate")
    builder.add_conditional_edges("assess_retry", route_after_retry, {"submit_training": "submit_training", "advance_candidate": "advance_candidate", "finish_failed": "finish_failed"})
    builder.add_conditional_edges("advance_candidate", route_after_advance, {"prepare_candidate": "prepare_candidate", "compare_final": "compare_final"})
    builder.add_edge("compare_final", END)
    builder.add_edge("finish_blocked", END)
    builder.add_edge("finish_cancelled", END)
    builder.add_edge("finish_failed", END)
    return builder.compile(checkpointer=checkpointer)
