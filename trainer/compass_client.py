import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ENDPOINT = "https://compass.llm.shopee.io/compass-api/v1"
NON_TEXT_MODELS = re.compile(r"chirp|nmt|byteplus|-vl-|tts|whisper|embed|image|video|dall-e|imagen|veo|sora|audio", re.IGNORECASE)
AVAILABLE_FILE = "compass-available.json"


class CompassError(Exception):
    def __init__(self, message: str = "", status: int | None = None):
        super().__init__(message)
        self.status = status  # HTTP status when Compass answered; None for transport failures.

    @property
    def definite(self) -> bool:
        # A 4xx other than 429 (e.g. "model not supported", "not found or no access") will fail again; anything else may pass later.
        return self.status is not None and 400 <= self.status < 500 and self.status != 429


def configured() -> bool:
    return bool(os.getenv("ADVISOR_API_KEY")) and os.getenv("ADVISOR_BASE_URL", "").rstrip("/") == ENDPOINT


def safe_message(message: Any) -> str:
    key = os.getenv("ADVISOR_API_KEY")
    text = str(message or "") or "Compass request failed"
    return (text.replace(key, "[redacted]") if key else text)[:500]


def request_compass(path: str, body: dict[str, Any] | None = None, *, timeout: float = 120, retries: int = 3) -> dict[str, Any]:
    if not configured():
        raise CompassError("Set ADVISOR_API_KEY and the Compass ADVISOR_BASE_URL in the trainer's .env, then restart the trainer.")
    headers = {"Authorization": "Bearer " + os.environ["ADVISOR_API_KEY"], "Content-Type": "application/json"}
    for attempt in range(retries):
        request = urllib.request.Request(ENDPOINT + path, data=json.dumps(body).encode() if body is not None else None, headers=headers, method="POST" if body is not None else "GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read())
        except urllib.error.HTTPError as error:
            if error.code == 429 and attempt < retries - 1:
                time.sleep(30)
                continue
            try:
                message = json.loads(error.read()).get("message")
            except (ValueError, AttributeError, OSError):
                message = None
            raise CompassError(safe_message(message or f"Compass returned HTTP {error.code}."), error.code) from None
        except Exception as error:  # Any transport or parse failure (e.g. IncompleteRead) must stay a redacted CompassError.
            raise CompassError(safe_message(error)) from None
        if not isinstance(data, dict):
            raise CompassError("Compass returned an unexpected response.")
        return data
    raise CompassError("Compass request failed")


def text_models() -> list[str]:
    data = request_compass("/models").get("data")
    ids = (item.get("id") for item in (data if isinstance(data, list) else []) if isinstance(item, dict))
    return sorted(model for model in ids if isinstance(model, str) and not NON_TEXT_MODELS.search(model))


def chat(model: str, messages: list[dict[str, Any]], max_tokens: int = 256, temperature: float = 0) -> dict[str, Any]:
    kept = [{"role": m["role"], "content": m["content"]} for m in messages if not (m["role"] == "system" and not str(m["content"]).strip())]
    if not kept or not str(kept[-1]["content"]).strip():
        raise CompassError("Conversation input is empty.")
    started = time.monotonic()
    data = request_compass("/chat/completions", {"model": model, "messages": kept, "temperature": temperature, "max_tokens": max_tokens})
    choices = data.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    message = choice.get("message")
    output = message.get("content") if isinstance(message, dict) else None
    if not isinstance(output, str) or not output.strip():
        raise CompassError("Compass returned no answer text.")
    return {"output": output, "finishReason": choice.get("finish_reason"), "usage": data.get("usage") or {}, "seconds": time.monotonic() - started}


def check_failure(model: str) -> CompassError | None:
    try:
        request_compass("/chat/completions", {"model": model, "messages": [{"role": "user", "content": "Reply with OK."}], "max_tokens": 5, "temperature": 0}, timeout=20, retries=1)
    except CompassError as error:
        return error
    return None


# The catalog lists models this key cannot run, so the whole catalog is checked once and the models that answered are saved.
# The saved list is reused until someone rechecks; a definite refusal found later takes that one model off it.
def load_available(data_root: Path) -> dict[str, Any] | None:
    try:
        saved = json.loads((Path(data_root) / AVAILABLE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None  # Never checked on this trainer, or the file is unreadable.
    return saved if isinstance(saved, dict) and isinstance(saved.get("models"), list) else None


def _write_available(data_root: Path, saved: dict[str, Any]) -> dict[str, Any]:
    (Path(data_root) / AVAILABLE_FILE).write_text(json.dumps(saved), encoding="utf-8")
    return saved


def save_available(data_root: Path, models: list[str]) -> dict[str, Any]:
    return _write_available(data_root, {"models": sorted(models), "checkedAt": datetime.now(timezone.utc).isoformat()})


def drop_available(data_root: Path, model: str) -> None:
    saved = load_available(data_root)
    if saved and model in saved["models"]:
        _write_available(data_root, {**saved, "models": [m for m in saved["models"] if m != model]})


def check_model(model: str) -> str | None:
    error = check_failure(model)
    return str(error) if error else None


def answer_cases(model: str, cases: list[dict[str, Any]], on_answer: Callable[[int, int], None] | None = None, max_tokens: int = 256) -> list[dict[str, Any]]:
    results = []
    for case in cases:
        answer = chat(model, case["messages"], max_tokens=max_tokens)
        results.append({"model": model, "rowId": case["rowId"], **answer})
        if on_answer:
            on_answer(len(results), len(cases))
    return results
