from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re


_RUN_DIR_PATTERN = re.compile(r"^log-(\d+)-(\d{2})-(\d{2})-(\d{4})$")


def allocate_run_log_dir(root: str | Path = "logs", *, now: datetime | None = None) -> Path:
    base_dir = Path(root)
    base_dir.mkdir(parents=True, exist_ok=True)

    highest_iteration = 0
    for entry in base_dir.iterdir():
        if not entry.is_dir():
            continue
        match = _RUN_DIR_PATTERN.match(entry.name)
        if match is None:
            continue
        highest_iteration = max(highest_iteration, int(match.group(1)))

    current_time = datetime.now() if now is None else now
    next_iteration = highest_iteration + 1
    folder_name = (
        f"log-{next_iteration}-"
        f"{current_time.month:02d}-{current_time.day:02d}-{current_time.year:04d}"
    )
    run_dir = base_dir / folder_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir
