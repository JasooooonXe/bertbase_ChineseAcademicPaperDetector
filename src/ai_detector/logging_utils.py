from __future__ import annotations

import io
import json
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

from ai_detector.config import dump_yaml_config


@dataclass
class RunArtifacts:
    run_dir: Path
    checkpoint_dir: Path
    stdout_path: Path
    stderr_path: Path
    config_path: Path
    metrics_path: Path
    metadata_path: Path


class TeeStream(io.TextIOBase):
    def __init__(self, *streams: io.TextIOBase) -> None:
        self._streams = streams

    def write(self, text: str) -> int:
        for stream in self._streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def get_git_commit() -> str:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"
    return output or "unknown"


def create_run_artifacts(output_root: str | Path, experiment_name: str) -> RunArtifacts:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{experiment_name}"
    output_root = Path(output_root)
    run_dir = output_root / "runs" / run_name
    checkpoint_dir = output_root / "checkpoints" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return RunArtifacts(
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        stdout_path=run_dir / "stdout.log",
        stderr_path=run_dir / "stderr.log",
        config_path=run_dir / "resolved_config.yaml",
        metrics_path=run_dir / "metrics.json",
        metadata_path=run_dir / "metadata.json",
    )


def save_run_metadata(
    artifacts: RunArtifacts,
    config: dict,
    extra_metadata: dict | None = None,
) -> None:
    dump_yaml_config(artifacts.config_path, config)
    metadata = {"git_commit": get_git_commit()}
    if extra_metadata:
        metadata.update(extra_metadata)
    with artifacts.metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


def save_metrics(path: str | Path, metrics: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)


@contextmanager
def tee_std_streams(
    stdout_path: str | Path,
    stderr_path: str | Path,
) -> Iterator[None]:
    stdout_path = Path(stdout_path)
    stderr_path = Path(stderr_path)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        stdout_path.open("a", encoding="utf-8") as stdout_file,
        stderr_path.open("a", encoding="utf-8") as stderr_file,
    ):
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = TeeStream(old_stdout, stdout_file)
        sys.stderr = TeeStream(old_stderr, stderr_file)
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
