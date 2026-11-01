"""Live progress for long workflow steps, written to a file the page polls."""

import json
import os
import time
from pathlib import Path


class ProgressFile:
    """Records how much of a step is done. Picklable, so isolated inference processes can report too.

    A step is measured in units (one answered or graded question each). This writer covers units
    base..base+count of total, so several phases can fill one bar.
    """

    def __init__(self, path: Path, message: str, base: int, total: int, started_at: float | None = None):
        self.path, self.message, self.base, self.total = Path(path), message, base, max(1, total)
        self.started_at = started_at or time.time()
        self(0, 0)

    def __call__(self, done: int, _count: int) -> None:
        percent = min(100, round(100 * (self.base + done) / self.total))
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps({"message": self.message, "startedAt": self.started_at, "percent": percent}), encoding="utf-8")
            os.replace(temporary, self.path)
        except OSError:
            pass  # Progress is best effort; it must never fail a workflow step.
