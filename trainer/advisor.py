"""Local-first dataset profiling and optional OpenAI-compatible cloud clarification."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from datasets import Dataset, load_dataset

MODELS = [
    {"id": "Qwen/Qwen2.5-3B-Instruct", "name": "Qwen 2.5", "sizeB": 3, "vram": 8, "license": "Qwen Research", "task": "general"},
    {"id": "Qwen/Qwen2.5-7B-Instruct", "name": "Qwen 2.5", "sizeB": 7, "vram": 13, "license": "Apache 2.0", "task": "general"},
    {"id": "Qwen/Qwen2.5-Coder-7B-Instruct", "name": "Qwen 2.5 Coder", "sizeB": 7, "vram": 13, "license": "Apache 2.0", "task": "code"},
    {"id": "mistralai/Mistral-7B-Instruct-v0.3", "name": "Mistral", "sizeB": 7, "vram": 13, "license": "Apache 2.0", "task": "english-writing"},
    {"id": "Qwen/Qwen2.5-14B-Instruct", "name": "Qwen 2.5", "sizeB": 14, "vram": 22, "license": "Apache 2.0", "task": "general"},
]
EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)")
SECRET = re.compile(r"\b(?:sk|api|key|token|secret)[_-]?[A-Za-z0-9_-]{12,}\b", re.I)
CREDENTIAL_URL = re.compile(r"https?://[^\s/@:]+:[^\s/@]+@", re.I)
CODE = re.compile(r"```(?:python|javascript|typescript|java|cpp|sql|bash)\b|^\s*(?:def\s+\w+\s*\(|class\s+\w+[^\n]*:|from\s+[\w.]+\s+import\s|import\s+[\w.]+\s*$|(?:const|let|var)\s+\w+\s*=|function\s+\w+\s*\(|SELECT\s+.+\s+FROM\s+)", re.I | re.M)


def normalize_columns(dataset: Dataset) -> Dataset:
    """Map Alpaca-style instruction/input/output rows to prompt/completion.

    Some datasets (e.g. Bitext customer support) name the answer `response`.
    Their `text` column repeats the whole example, answer included, so it is
    dropped rather than used as model input.
    """
    columns = set(dataset.column_names)
    answer = "output" if "output" in columns else "response"
    if not {"instruction", answer} <= columns or columns & {"prompt", "completion", "messages"}:
        return dataset
    def pair(row):
        extra = str(row.get("input") or "").strip()
        return {"prompt": str(row["instruction"]) + ("\n\n" + extra if extra else ""), "completion": str(row[answer])}
    return dataset.map(pair, remove_columns=[name for name in ("instruction", "input", answer, "text") if name in columns])


def load_source(source: str, name: str, extension: str | None = None) -> Dataset:
    if source == "huggingface":
        return normalize_columns(load_dataset(name, split="train"))
    suffix = (extension or Path(name).suffix.lstrip(".")).lower()
    loader = {"csv": "csv", "jsonl": "json", "parquet": "parquet"}.get(suffix)
    if not loader:
        raise ValueError("Supported file types are CSV, JSONL, and Parquet.")
    return normalize_columns(load_dataset(loader, data_files=name, split="train"))


def content_for_row(row: dict[str, Any]) -> str:
    if isinstance(row.get("messages"), list):
        return "\n".join(str(item.get("content", "")) if isinstance(item, dict) else str(item) for item in row["messages"])
    if "prompt" in row or "completion" in row:
        return str(row.get("prompt", "")) + "\n" + str(row.get("completion", ""))
    return str(row.get("text", ""))


def redact(text: str) -> tuple[str, int]:
    before = text
    text = EMAIL.sub("[EMAIL]", text)
    text = PHONE.sub("[PHONE]", text)
    text = SECRET.sub("[SECRET]", text)
    text = CREDENTIAL_URL.sub("https://[CREDENTIALS]@", text)
    return text, int(text != before)


def input_key(row):
    """Group repeated inputs even when their target answers differ."""
    value = row.get("prompt", row.get("text", ""))
    if isinstance(row.get("messages"), list):
        messages = row["messages"]
        value = json.dumps(messages[:-1] if messages and messages[-1].get("role") == "assistant" else messages, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(" ".join(str(value).casefold().split()).encode()).hexdigest()


def privacy_scan(rows: list[dict[str, Any]], limit: int = 600) -> dict[str, Any]:
    """Scan a bounded sample for values that should not leave the local app."""
    patterns = {
        "email": EMAIL,
        "phone": PHONE,
        "secret": SECRET,
        "credentialUrl": CREDENTIAL_URL,
    }
    matches = {name: 0 for name in patterns}
    flagged_rows: list[int] = []
    sampled = list(rows)[:limit]
    for row_id, row in enumerate(sampled):
        text = content_for_row(row)
        flagged = False
        for name, pattern in patterns.items():
            count = len(pattern.findall(text))
            matches[name] += count
            flagged = flagged or count > 0
        if flagged:
            flagged_rows.append(row_id)
    flagged_count = len(flagged_rows)
    return {
        "status": "review" if flagged_count else "clear",
        "safeToSendToHostedModel": not flagged_count,
        "rowsScanned": len(sampled),
        "flaggedRows": flagged_rows[:100],
        "flaggedRowCount": flagged_count,
        "matches": matches,
        "message": "Sensitive values detected; review or redact before hosted model comparisons." if flagged_count else "No common email, phone, secret, or credential URL patterns detected in the sample.",
    }


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    return values[min(len(values) - 1, max(0, round((len(values) - 1) * fraction)))]


def detect_schema(columns: list[str]) -> str | None:
    if "text" in columns and any(name in columns for name in ("label", "class", "target")):
        return "classification"
    if "messages" in columns:
        return "chat"
    if {"prompt", "completion"}.issubset(columns):
        return "prompt_completion"
    if "text" in columns:
        return "text"
    return None


def classification_label_field(columns: list[str]) -> str | None:
    return next((name for name in ("label", "class", "target") if name in columns), None)


def choose_positive_label(labels: list[str]) -> str | None:
    """Choose a transparent default positive class when the user has not set one."""
    normalized = {label.lower(): label for label in labels}
    for preferred in ("1", "true", "yes", "positive", "spam", "fraud", "unsafe"):
        if preferred in normalized:
            return normalized[preferred]
    return labels[-1] if len(labels) == 2 else None


def classification_report(rows: list[dict[str, Any]], label_field: str, predictions: list[str]) -> dict[str, Any]:
    """Create reproducible class metrics and an error queue without an LLM judge."""
    if len(rows) != len(predictions):
        raise ValueError("Each labelled row needs one prediction.")
    labels = sorted({str(row.get(label_field, "")).strip() for row in rows if str(row.get(label_field, "")).strip()})
    positive = choose_positive_label(labels)
    tp = fp = fn = tn = 0
    errors: list[dict[str, Any]] = []
    correct = 0
    for row, predicted in zip(rows, predictions):
        expected = str(row.get(label_field, "")).strip()
        predicted = str(predicted).strip()
        if predicted == expected:
            correct += 1
        if positive:
            if expected == positive and predicted == positive:
                tp += 1
            elif expected != positive and predicted == positive:
                fp += 1
            elif expected == positive and predicted != positive:
                fn += 1
            else:
                tn += 1
        if predicted != expected:
            category = "false_negative" if positive and expected == positive else "false_positive" if positive and predicted == positive else "wrong_class"
            errors.append({
                "rowId": row.get("_row_id"), "input": str(row.get("text", ""))[:600],
                "expected": expected, "predicted": predicted or "(no recognized label)", "category": category,
            })
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return {
        "supported": True, "labelField": label_field, "labels": labels, "positiveLabel": positive,
        "total": len(rows), "correct": correct, "accuracy": round(correct / len(rows), 4) if rows else None,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn} if positive else None,
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
        "errors": errors[:100], "errorCount": len(errors),
    }


def classification_prediction(output: str, labels: list[str]) -> str:
    normalized = re.sub(r"^label\s*:\s*", "", output.strip(), flags=re.I).strip(" .\"'")
    return next((label for label in labels if label.casefold() == normalized.casefold()), "")


def task_quality(rows, references, outputs, row_ids):
    """Measure labelled or structured targets; abstain on free-form correctness."""
    rows = list(rows)
    if not (len(rows) == len(references) == len(outputs) == len(row_ids)):
        raise ValueError("Quality evaluation inputs must have matching lengths.")
    field = classification_label_field(list(rows[0])) if rows else None
    evidence = {"metric": None, "score": None, "coverage": 0, "evaluatedExamples": 0, "totalExamples": len(rows)}
    if field and all("text" in row and field in row for row in rows):
        labels = sorted({str(row[field]).strip() for row in rows})
        report = classification_report([{**row, "_row_id": i} for row, i in zip(rows, row_ids)], field, [classification_prediction(output, labels) for output in outputs])
        evidence.update(metric="label_accuracy", score=report["accuracy"], coverage=1, evaluatedExamples=len(rows))
        return {"qualityEvidence": evidence, "classification": report}
    scores = []
    for reference, output in zip(references, outputs):
        try:
            expected = json.loads(reference)
            if not isinstance(expected, (dict, list)):
                continue
        except (TypeError, ValueError):
            continue
        try:
            actual = json.loads(output)
            scores.append(int(json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)))
        except (TypeError, ValueError):
            scores.append(0)
    if scores:
        evidence.update(metric="json_exact_match", score=sum(scores)/len(scores), coverage=len(scores)/len(rows), evaluatedExamples=len(scores))
    return {"qualityEvidence": evidence}


def local_profile(dataset: Dataset) -> dict[str, Any]:
    columns = list(dataset.column_names)
    schema = detect_schema(columns)
    sample_size = min(len(dataset), 600)
    indices = sorted(random.Random(42).sample(range(len(dataset)), sample_size))
    sample = dataset.select(indices)
    valid, invalid, texts, language = [], [], [], Counter()
    duplicates, code_count, redactions = 0, 0, 0
    seen: set[str] = set()
    samples: list[dict[str, Any]] = []
    for index, row in zip(indices, sample):
        text = content_for_row(row).strip()
        if not text or not schema:
            invalid.append(index)
            continue
        digest = hashlib.sha256(re.sub(r"\s+", " ", text.lower()).encode()).hexdigest()
        duplicates += digest in seen
        seen.add(digest)
        cjk = sum("\u4e00" <= ch <= "\u9fff" for ch in text)
        alpha = sum(ch.isalpha() for ch in text)
        language["Chinese / CJK" if cjk > 3 else "English / Latin" if alpha > 3 else "Unknown"] += 1
        code_count += bool(CODE.search(text))
        clean, count = redact(text)
        redactions += count
        valid.append(index)
        texts.append(text)
        if len(samples) < 20:
            samples.append({"rowId": index, "text": clean[:1200]})

    lengths = [len(value) for value in texts]
    duplicate_rate = duplicates / max(1, len(valid))
    invalid_rate = len(invalid) / max(1, sample_size)
    code_ratio = code_count / max(1, len(valid))
    task = "classification" if schema == "classification" else "code" if code_ratio > .22 else "assistant" if schema == "chat" else "structured" if any("{" in value and "}" in value for value in texts[:30]) else "writing"
    confidence = .95 if schema == "classification" else .92 if schema in {"chat", "prompt_completion"} else .74 if schema == "text" else .25
    findings = []
    if not schema:
        findings.append({"severity": "blocker", "code": "unsupported_schema", "message": "Need messages, prompt/completion, or text columns."})
    if len(valid) < 30:
        findings.append({"severity": "blocker", "code": "too_small", "message": "Fewer than 30 valid examples. Use prompting/RAG or collect more examples."})
    elif len(valid) < 500:
        findings.append({"severity": "warning", "code": "small_dataset", "message": "Fewer than 500 examples; use low rank and review overfitting carefully."})
    if invalid_rate > .2:
        findings.append({"severity": "blocker", "code": "invalid_rows", "message": "More than 20% of sampled rows are empty or incompatible."})
    if duplicate_rate > .25:
        findings.append({"severity": "blocker", "code": "duplicates", "message": "More than 25% of sampled rows are duplicates; deduplicate before training."})
    if task == "writing" and len(valid) < 100:
        findings.append({"severity": "warning", "code": "uncertain_task", "message": "Task classification is uncertain; cloud clarification is available if enabled."})
    return {
        "facts": {
            "schema": schema or "unknown", "columns": columns, "labelField": classification_label_field(columns), "rows": len(dataset), "sampledRows": sample_size, "sampling": "seeded random sample across all rows (seed 42)",
            "validRows": len(valid), "rejectedRows": len(invalid), "rejectedRowIds": invalid[:50],
            "languages": dict(language), "codeRatio": round(code_ratio, 3), "duplicateRate": round(duplicate_rate, 3),
            "lengthCharacters": {"p50": percentile(lengths, .5), "p95": percentile(lengths, .95), "max": max(lengths, default=0)},
            "estimatedTokenLength": {"p50": math.ceil(percentile(lengths, .5) / 3.5), "p95": math.ceil(percentile(lengths, .95) / 3.5)},
        },
        "classification": {"task": task, "confidence": confidence, "source": "local"},
        "findings": findings,
        "redaction": {"sampleRows": len(samples), "fieldsRedacted": redactions, "sentToCloud": False},
        "_samples": samples,
    }


def rank_candidates(profile: dict[str, Any], goal: str, memory_gb: int = 24) -> list[dict[str, Any]]:
    task = profile["classification"]["task"]
    candidates = []
    for model in MODELS:
        method = "qlora" if model["sizeB"] >= 7 else ("lora" if goal == "quality" else "qlora")
        reason, rejected = [], None
        if model["vram"] > memory_gb:
            rejected = "Estimated VRAM exceeds available GPU memory."
        elif model["sizeB"] == 14 and method != "qlora":
            rejected = "14B requires QLoRA."
        elif task == "code" and model["task"] != "code":
            reason.append("General model; code-specialized candidate ranks higher.")
        elif task == "code":
            reason.append("Code-specialized instruction model.")
        elif task == "assistant" and model["id"].startswith("Qwen"):
            reason.append("Strong multilingual instruction fit.")
        elif goal == "speed" and model["sizeB"] == 3:
            reason.append("Smallest eligible model for fast iteration.")
        elif model["sizeB"] == 14:
            reason.append("Higher capacity but slower; use only if baseline improves materially.")
        score = (100 - model["sizeB"] * (9 if goal == "speed" else 2)) + (30 if task == "code" and model["task"] == "code" else 0) + (16 if task == "assistant" and model["id"].startswith("Qwen") else 0)
        candidates.append({"model": model, "method": method, "estimatedVramGb": model["vram"], "estimatedDuration": "longer" if model["sizeB"] == 14 else "moderate" if model["sizeB"] == 7 else "short", "score": score, "rejected": rejected, "reasons": reason})
    return sorted(candidates, key=lambda item: (item["rejected"] is not None, -item["score"]))


def cloud_status() -> dict[str, Any]:
    configured = all(os.getenv(key) for key in ("ADVISOR_BASE_URL", "ADVISOR_API_KEY", "ADVISOR_MODEL"))
    return {"configured": configured, "baseUrl": os.getenv("ADVISOR_BASE_URL", "") if configured else None, "model": os.getenv("ADVISOR_MODEL", "") if configured else None}


def cloud_clarify(samples: list[dict[str, Any]], local: dict[str, Any]) -> dict[str, Any] | None:
    if not cloud_status()["configured"]:
        return None
    endpoint = os.environ["ADVISOR_BASE_URL"].rstrip("/") + "/chat/completions"
    prompt = {"localClassification": local["classification"], "facts": local["facts"], "rows": samples[:20], "instruction": "Return JSON only with task, languageSummary, confidence, and explanation. Do not infer personal data."}
    request = urllib.request.Request(endpoint, data=json.dumps({"model": os.environ["ADVISOR_MODEL"], "messages": [{"role": "user", "content": json.dumps(prompt)}], "temperature": 0}).encode(), headers={"Authorization": "Bearer " + os.environ["ADVISOR_API_KEY"], "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            content = json.loads(response.read())["choices"][0]["message"]["content"]
        return json.loads(content.strip())
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError, ValueError, json.JSONDecodeError):
        return None


def cloud_assess_blockers(samples: list[dict[str, Any]], profile: dict[str, Any], blockers: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Ask the optional cloud assistant for advice only; it cannot mutate data."""
    if not cloud_status()["configured"]:
        return None
    endpoint = os.environ["ADVISOR_BASE_URL"].rstrip("/") + "/chat/completions"
    prompt = {
        "profile": {"facts": profile.get("facts", {}), "classification": profile.get("classification", {})},
        "blockers": blockers,
        "redactedRows": samples[:20],
        "instruction": "Return JSON only: {canResolve:boolean, requiresHumanApproval:boolean, reason:string, suggestedOperations:string[]}. Give advice only. Never claim to have changed data, and never recommend bypassing a blocker.",
    }
    request = urllib.request.Request(endpoint, data=json.dumps({"model": os.environ["ADVISOR_MODEL"], "messages": [{"role": "user", "content": json.dumps(prompt)}], "temperature": 0}).encode(), headers={"Authorization": "Bearer " + os.environ["ADVISOR_API_KEY"], "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            content = json.loads(response.read())["choices"][0]["message"]["content"]
        result = json.loads(content.strip())
        return result if isinstance(result, dict) else None
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError, ValueError, json.JSONDecodeError):
        return None


def connection_test() -> dict[str, Any]:
    if not cloud_status()["configured"]:
        return {"ok": False, "message": "Cloud assist is not configured."}
    result = cloud_clarify([{"rowId": 0, "text": "Classify this harmless sample."}], {"classification": {"task": "unknown", "confidence": 0}, "facts": {}})
    return {"ok": result is not None, "message": "Connection succeeded." if result else "Connection failed or returned an unsupported response."}


def evaluation_messages(row, instruction=""):
    """Keep the final answer out of generation input for both model variants."""
    label_field = classification_label_field(list(row))
    reference = str(row[label_field]) if "text" in row and label_field else str(row.get("completion", ""))
    if isinstance(row.get("messages"), list):
        messages = [dict(message) for message in row["messages"]]
        if messages and messages[-1].get("role") == "assistant":
            reference = messages.pop().get("content", "")
        if not messages:
            raise ValueError("An evaluation example needs an input before its answer.")
        if instruction:
            messages[-1]["content"] = instruction + "\n\n" + str(messages[-1].get("content", ""))
        return messages, reference
    prompt = ("Text: " + str(row["text"]) + "\nLabel:") if "text" in row and label_field else str(row.get("prompt", row.get("text", "")))
    if instruction:
        prompt = instruction + "\n\n" + prompt
    return [{"role": "user", "content": prompt}], reference


def evaluation_prompt(row, tokenizer, instruction=""):
    messages, reference = evaluation_messages(row, instruction)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), reference


def retrieval_evaluation_prompt(row, tokenizer, instruction, documents, max_length=4096):
    from quality_probe import retrieval_instruction, retrieve, query_for
    selected = retrieve(query_for(row), documents) if documents else []
    while selected:
        enriched, sources = retrieval_instruction(row, instruction, selected)
        prompt, reference = evaluation_prompt(row, tokenizer, enriched)
        if len(tokenizer.encode(prompt, add_special_tokens=False)) <= max_length:
            return prompt, reference, sources
        # ponytail: keep document prefixes; use passage retrieval for larger corpora.
        selected = [{**doc, "text":doc["text"][:len(doc["text"]) // 2]} for doc in selected if len(doc["text"]) > 32]
    prompt, reference = evaluation_prompt(row, tokenizer, instruction)
    return prompt, reference, []


# Repetition shorter than this is normal (matrix rows, "& 0 & 0"); a loop repeats one block back to back for longer.
LOOP_TOKENS = 200


def _repeating(tokens: list[int]) -> bool:
    """True when the tokens end in one block repeated back to back over at least LOOP_TOKENS tokens (at least 3 copies)."""
    for period in range(1, LOOP_TOKENS + 1):
        copies = max(3, -(-LOOP_TOKENS // period))
        if len(tokens) < period * copies:
            continue
        block = tokens[-period:]
        if all(tokens[len(tokens) - (k + 1) * period:len(tokens) - k * period] == block for k in range(1, copies)):
            return True
    return False


# Measured on 250 saved Qwen answers: one that kept re-solving its question ("Solution 2" ... "Solution 27", never
# repeating a block exactly) had 97-100% of its last 2,048 tokens made of 32-token phrases it had already written;
# every other answer stayed at or below 56% (row-by-row trial and error is the highest).
PHRASE_TOKENS, REWRITE_WINDOW, REWRITE_SHARE = 32, 2048, 0.8


def _rewriting(tokens: list[int]) -> bool:
    """True when most of the last REWRITE_WINDOW tokens restate 32-token phrases written earlier in the answer."""
    start = len(tokens) - PHRASE_TOKENS + 1 - REWRITE_WINDOW  # First phrase position inside the window.
    if start < 0:
        return False
    seen, repeated = set(), 0
    for position in range(len(tokens) - PHRASE_TOKENS + 1):
        phrase = tuple(tokens[position:position + PHRASE_TOKENS])
        repeated += position >= start and phrase in seen
        seen.add(phrase)
    return repeated >= REWRITE_SHARE * REWRITE_WINDOW


def _looping(tokens: list[int]) -> bool:
    return _repeating(tokens) or _rewriting(tokens)


# Measured on the VM's T4s with 4-bit Qwen2.5-7B and Mistral-7B: 8, 16 and 32 answers at once gave about
# 65, 100 and 135 tokens/s; 48 ran out of memory even with 330-token prompts.
ANSWER_SLOTS = 32
# Answers have no length cap; this is only the growth each answer's KV cache is planned for when it joins.
ANSWER_ROOM = 4096


def _kv_bytes(model: Any) -> int:
    """float16 key and value bytes one token adds to the cache."""
    config = getattr(model, "config", None)
    heads = getattr(config, "num_attention_heads", None)
    head_dim = getattr(config, "head_dim", None) or (getattr(config, "hidden_size", 0) // heads if heads else 0)
    return 2 * getattr(config, "num_hidden_layers", 0) * getattr(config, "num_key_value_heads", heads or 0) * head_dim * 2


def _answer_stream(model: Any, tokenizer: Any, prompts: list[str], pending: Any, lock: Any, answers: list[Any], torch: Any,
                   max_new_tokens: int | None = None, on_answer: Callable[[], None] | None = None) -> None:
    """Greedy answers that join and leave one running batch: when an answer ends, the next question takes its slot.

    A batch that waits for its longest answer spends most of its time decoding one or two answers while the rest are
    finished; here the GPU keeps answering up to ANSWER_SLOTS questions at once until the queue is empty. Questions are
    taken from `pending` (shared under `lock` when several models answer), each answer is written to `answers`.
    Without max_new_tokens an answer runs until the model ends it, its context is full, or it loops."""
    ends = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    ends = {tokenizer.eos_token_id, *(ends if isinstance(ends, list) else [ends] if ends is not None else [])}
    pad = tokenizer.pad_token_id if getattr(tokenizer, "pad_token_id", None) is not None else tokenizer.eos_token_id
    context = getattr(getattr(model, "config", None), "max_position_embeddings", None) or 32768
    device = model.device
    budget = 0.8 * torch.cuda.mem_get_info(device)[0] if str(device).startswith("cuda") else float("inf")
    kv, slots = _kv_bytes(model), ANSWER_SLOTS
    rows: list[dict[str, Any]] = []
    cache = mask = upcoming = None

    def finish(row: dict[str, Any], outcome: str | None) -> None:
        answers[row["position"]] = {"text": tokenizer.decode(row["tokens"], skip_special_tokens=True),
                                    "hitTokenLimit": outcome == "limit", "stoppedRepeating": outcome == "loop"}
        with lock:
            if on_answer:
                on_answer()

    while True:
        joining = []
        width = mask.shape[1] if mask is not None else 0
        while len(rows) + len(joining) < slots:
            with lock:
                if not pending:
                    break
                ids = tokenizer(prompts[pending[0]], truncation=True, max_length=4096)["input_ids"]
                width = max(width, len(ids))
                if (rows or joining) and (len(rows) + len(joining) + 1) * (width + ANSWER_ROOM) * kv > budget:
                    break  # Joins once running answers end and free their cache.
                joining.append({"position": pending.popleft(), "ids": ids, "tokens": [], "limit": min(max_new_tokens or context, context - len(ids))})
        if not rows and not joining:
            return
        out_of_memory = False
        try:
            with torch.no_grad():
                if joining:
                    longest = max(len(row["ids"]) for row in joining)
                    ids = torch.tensor([[pad] * (longest - len(row["ids"])) + row["ids"] for row in joining], device=device)
                    joined = torch.tensor([[0] * (longest - len(row["ids"])) + [1] * len(row["ids"]) for row in joining], device=device)
                    # Only the last position's scores are needed; all of them (prompt x 152k-word vocabulary) ran 32 prompts out of memory.
                    out = model(input_ids=ids, attention_mask=joined, position_ids=(joined.cumsum(-1) - 1).clamp(min=0), use_cache=True, logits_to_keep=1)
                    first = out.logits[:, -1].argmax(-1)
                    if cache is None:
                        cache, mask, upcoming = out.past_key_values, joined, first
                    else:
                        # Left-pad the shorter side so every row's cache ends at its latest token; padding is masked out.
                        size = max(mask.shape[1], longest)
                        grow = lambda tensor, have: torch.nn.functional.pad(tensor, (0, 0, size - have, 0))
                        for layer, new in zip(cache.layers, out.past_key_values.layers):
                            layer.keys = torch.cat([grow(layer.keys, mask.shape[1]), grow(new.keys, longest)])
                            layer.values = torch.cat([grow(layer.values, mask.shape[1]), grow(new.values, longest)])
                        mask = torch.cat([torch.nn.functional.pad(mask, (size - mask.shape[1], 0)), torch.nn.functional.pad(joined, (size - longest, 0))])
                        upcoming = torch.cat([upcoming, first])
                    out = ids = joined = first = None
                    rows += joining
                    joining = []
                keep = []
                for index, (row, token) in enumerate(zip(rows, upcoming.tolist())):
                    outcome = "ended" if token in ends else None
                    if not outcome:
                        row["tokens"].append(token)
                        count = len(row["tokens"])
                        # Exact repeats are checked every 32 tokens; the costlier rewriting check every 256.
                        if (count % 32 == 0 and _repeating(row["tokens"][-3 * LOOP_TOKENS:])) or (count % 256 == 0 and count >= REWRITE_WINDOW + PHRASE_TOKENS and _rewriting(row["tokens"])):
                            outcome = "loop"
                        elif count >= row["limit"]:
                            outcome = "limit"
                    if outcome:
                        finish(row, outcome)
                    else:
                        keep.append(index)
                if len(keep) < len(rows):
                    rows = [rows[index] for index in keep]
                    if not rows:
                        cache = mask = upcoming = None
                        continue
                    chosen = torch.tensor(keep, device=device)
                    cache.batch_select_indices(chosen)
                    mask, upcoming = mask[chosen], upcoming[chosen]
                    cut = int(mask.any(0).int().argmax())  # Columns that are padding in every remaining row.
                    if cut:
                        mask = mask[:, cut:]
                        for layer in cache.layers:
                            layer.keys, layer.values = layer.keys[:, :, cut:], layer.values[:, :, cut:]
                mask = torch.cat([mask, mask.new_ones(len(rows), 1)], dim=1)
                out = model(input_ids=upcoming[:, None], attention_mask=mask, position_ids=mask.sum(-1, keepdim=True) - 1, past_key_values=cache, use_cache=True)
                cache, upcoming, out = out.past_key_values, out.logits[:, -1].argmax(-1), None
        except torch.cuda.OutOfMemoryError:
            if len(rows) + len(joining) <= 1:
                raise
            out_of_memory = True
        if out_of_memory:
            # The failed step may have grown some layers' caches and not others: start the running answers again, fewer at once.
            slots = max(1, (len(rows) + len(joining)) // 2)
            with lock:
                pending.extendleft(reversed([row["position"] for row in rows + joining]))
            rows, joining, cache, mask, upcoming, out = [], [], None, None, None, None
            gc.collect()
            torch.cuda.empty_cache()


def answer_models(load: Callable[[Any], Any], torch: Any) -> list[Any]:
    """One copy of the model per GPU that holds it alone, all answering from one queue; a model too big for one GPU
    is spread over them by device_map="auto". `load` takes a device_map."""
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if count < 2:
        return [load("auto")]
    devices = sorted(range(count), key=lambda device: -torch.cuda.mem_get_info(device)[0])
    models = None
    try:
        models = [load({"": devices[0]})]
    except torch.cuda.OutOfMemoryError:
        pass
    if models is None:
        gc.collect()
        torch.cuda.empty_cache()
        return [load("auto")]
    # Each copy also needs room for its answers' cache; another user's job may hold part of a GPU.
    footprint = models[0].get_memory_footprint()
    models += [load({"": device}) for device in devices[1:] if torch.cuda.mem_get_info(device)[0] > 2 * footprint]
    return models


def _answer_all(models: list[Any], tokenizer: Any, prompts: list[str], torch: Any, max_new_tokens: int | None = None,
                on_answer: Callable[[int], None] | None = None) -> tuple[list[dict[str, Any]], list[float]]:
    """Answers every prompt, each model on its own thread; returns the answers in prompt order and each one's share of
    the time (answers overlap, so per-answer time is not measured)."""
    import threading
    from collections import deque

    # Shortest prompts first: every running answer is padded to the longest one in the batch, and on a T4 padded attention
    # copies the cache once per query head (8x for Qwen2.5), so a 4,096-token prompt slowed 15 others from 10.7 to 2.8 steps/s.
    order = sorted(range(len(prompts)), key=lambda i: len(tokenizer(prompts[i], truncation=True, max_length=4096)["input_ids"]))
    pending, lock, answers, errors = deque(order), threading.Lock(), [None] * len(prompts), []
    report = (lambda: on_answer(sum(answer is not None for answer in answers))) if on_answer else None

    def work(model: Any) -> None:
        try:
            _answer_stream(model, tokenizer, prompts, pending, lock, answers, torch, max_new_tokens, report)
        except Exception as error:  # Re-raised below; the other copies finish the queue.
            errors.append(error)

    started = time.perf_counter()
    threads = [threading.Thread(target=work, args=(model,)) for model in models[1:]]
    for thread in threads:
        thread.start()
    work(models[0])
    for thread in threads:
        thread.join()
    if errors and any(answer is None for answer in answers):
        raise errors[0]
    share = (time.perf_counter() - started) / max(1, len(prompts))
    return answers, [share] * len(prompts)


def baseline_compare(
    dataset: Dataset,
    candidates: list[dict[str, Any]],
    limit: int = 20,
    eval_indices: list[int] | None = None,
    task: str | None = None,
    instruction: str = "",
    knowledge_documents: list[dict[str, Any]] | None = None,
    max_new_tokens: int | None = None,
    strict_context: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    """Run small local base-model comparisons. This intentionally loads one model at a time."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch

    selected_indices = list(eval_indices)[:limit] if eval_indices is not None else list(range(len(dataset)))[:limit]
    rows = dataset.select(selected_indices)
    results = []
    for number, candidate in enumerate(candidates):
        if on_progress:
            on_progress(number * len(rows), len(candidates) * len(rows))
        started = time.perf_counter()
        model = models = inputs = loss_output = output = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(candidate["model"]["id"], use_fast=True)
            tokenizer.pad_token = tokenizer.eos_token
            # Batched generation must continue each prompt from its own last token.
            tokenizer.padding_side = "left"
            # Load as training does (train.py): T4s compute far faster in float16 than in the float32 default.
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16) if candidate["method"] == "qlora" else None
            models = answer_models(lambda device_map: AutoModelForCausalLM.from_pretrained(candidate["model"]["id"], quantization_config=quant, device_map=device_map, torch_dtype=torch.float16), torch)
            model = models[0]
            samples, scores, format_scores, losses = [], [], [], []
            answer_seconds = []
            references, predictions = [], []
            prepared = []
            for index, row in enumerate(rows):
                prompt, reference, retrieved = retrieval_evaluation_prompt(row, tokenizer, instruction, knowledge_documents)
                length = len(tokenizer.encode(prompt))
                if strict_context and length > 4096:
                    raise ValueError("Benchmark input exceeds the 4096-token evaluation window; no truncated comparison is allowed.")
                prepared.append((index, prompt, reference, retrieved, length))
            for _, prompt, *_ in prepared:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
                with torch.no_grad():
                    loss_output = model(**inputs, labels=inputs["input_ids"])
                    if loss_output.loss is not None and math.isfinite(float(loss_output.loss.item())):
                        losses.append(float(loss_output.loss.item()))
            inputs = loss_output = None
            report = (lambda done, number=number: on_progress(number * len(rows) + done, len(candidates) * len(rows))) if on_progress else None
            answers, answer_seconds = _answer_all(models, tokenizer, [item[1] for item in prepared], torch, max_new_tokens, report)
            for (index, prompt, reference, retrieved, length), answer in zip(prepared, answers):
                generated = answer["text"]
                references.append(reference)
                predictions.append(generated)
                if reference:
                    expected = set(re.findall(r"\w+", reference.lower()))
                    actual = set(re.findall(r"\w+", generated.lower()))
                    scores.append(len(expected & actual) / max(1, len(expected | actual)))
                if task == "structured":
                    try:
                        json.loads(generated)
                        format_scores.append(1.0)
                    except (TypeError, json.JSONDecodeError):
                        format_scores.append(0.0)
                samples.append({"retrieved": retrieved, "rowId": selected_indices[index], "input": prompt, "reference": reference, "baseOutput": generated, "referenceAvailable": bool(reference), "inputTruncated": length > 4096, "hitTokenLimit": answer["hitTokenLimit"], "stoppedRepeating": answer["stoppedRepeating"]})
            eval_loss = round(sum(losses) / len(losses), 4) if losses else None
            results.append({**task_quality(list(rows), references, predictions, selected_indices), "modelId": candidate["model"]["id"], "status": "complete", "examples": len(rows), "evaluationIds": selected_indices, "eval_loss": eval_loss, "perplexity": round(math.exp(eval_loss), 3) if eval_loss is not None and eval_loss < 700 else None, "meanTokenOverlap": round(sum(scores) / len(scores), 3) if scores else None, "formatValidRate": round(sum(format_scores) / len(format_scores), 3) if format_scores else None, "inferenceSeconds": sum(answer_seconds), "meanLatencySeconds": sum(answer_seconds) / len(answer_seconds) if answer_seconds else None, "maxOutputTokens": max_new_tokens, "latencySeconds": round(time.perf_counter() - started, 1), "samples": samples})
        except Exception as error:
            results.append({"modelId": candidate["model"]["id"], "status": "unavailable", "message": str(error)[:300]})
        finally:
            # Release tensors and model hook cycles before freeing the CUDA allocator cache.
            model = models = inputs = loss_output = output = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return results


def evaluate_tuned_run(
    dataset: Dataset,
    candidate: dict[str, Any],
    run_dir: Path,
    eval_indices: list[int],
    test_dataset: Dataset | None = None,
    task: str | None = None,
    instruction: str = "",
    knowledge_documents: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate a saved adapter on the same deterministic examples as its baseline."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch

    model_id = candidate["model"]["id"]
    method = candidate.get("method", "qlora")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    # Same precision and batched, unlimited generation as the baseline comparison, so base and tuned answers are compared fairly.
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16) if method == "qlora" else None
    model = models = inputs = loss_output = output = None
    try:
        models = answer_models(lambda device_map: PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization, device_map=device_map, torch_dtype=torch.float16), str(run_dir / "adapter")), torch)
        model = models[0]
        rows_dataset = test_dataset if test_dataset is not None else dataset
        rows = rows_dataset.select(list(eval_indices))
        scores: list[float] = []
        format_scores: list[float] = []
        losses: list[float] = []
        samples: list[dict[str, Any]] = []
        references, predictions = [], []
        started = time.perf_counter()
        prepared = [retrieval_evaluation_prompt(row, tokenizer, instruction, knowledge_documents) for row in rows]
        for prompt, _, _ in prepared:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
            with torch.no_grad():
                loss_output = model(**inputs, labels=inputs["input_ids"])
                if loss_output.loss is not None and math.isfinite(float(loss_output.loss.item())):
                    losses.append(float(loss_output.loss.item()))
        inputs = loss_output = None
        lengths = [len(tokenizer.encode(prompt)) for prompt, _, _ in prepared]
        answers, _ = _answer_all(models, tokenizer, [prompt for prompt, _, _ in prepared], torch)
        for position, ((prompt, reference, retrieved), answer) in enumerate(zip(prepared, answers)):
            generated = answer["text"]
            references.append(reference)
            predictions.append(generated)
            if reference:
                expected = set(re.findall(r"\w+", reference.lower()))
                actual = set(re.findall(r"\w+", generated.lower()))
                scores.append(len(expected & actual) / max(1, len(expected | actual)))
            if task == "structured":
                try:
                    json.loads(generated)
                    format_scores.append(1.0)
                except (TypeError, json.JSONDecodeError):
                    format_scores.append(0.0)
            if position < 100:
                samples.append({"input": prompt, "retrieved": retrieved, "rowId": eval_indices[position], "tunedOutput": generated, "referenceAvailable": bool(reference), "inputTruncated": lengths[position] > 4096, "hitTokenLimit": answer["hitTokenLimit"], "stoppedRepeating": answer["stoppedRepeating"]})
        eval_loss = round(sum(losses) / len(losses), 4) if losses else None
        result = {
            **task_quality(list(rows), references, predictions, list(eval_indices)),
            "status": "complete",
            "modelId": model_id,
            "examples": len(rows),
            "evaluationIds": list(eval_indices),
            "eval_loss": eval_loss,
            "perplexity": round(math.exp(eval_loss), 3) if eval_loss is not None and eval_loss < 700 else None,
            "meanTokenOverlap": round(sum(scores) / len(scores), 3) if scores else None,
            "formatValidRate": round(sum(format_scores) / len(format_scores), 3) if format_scores else None,
            "latencySeconds": round(time.perf_counter() - started, 1),
            "samples": samples,
        }
        return result
    finally:
        model = models = inputs = loss_output = output = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
