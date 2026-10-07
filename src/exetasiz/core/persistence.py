"""Structured persistence, atomic writes, and run loading for Exetasiz."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml
from pydantic import ValidationError

from exetasiz.core.models import (
    AggregateMetric,
    DatasetMetadata,
    EvalResult,
    LoadedRun,
    ResultsEnvelope,
    StructuredMetrics,
)

SUPPORTED_SCHEMA_VERSIONS = {"1.0"}


class RunLoadError(Exception):
    """Raised when an evaluation run directory cannot be loaded or is invalid."""


class UnsupportedSchemaVersionError(RunLoadError):
    """Raised when the schema_version of results.json is not supported."""


def atomic_write_file(target_path: Union[str, Path], content: str, encoding: str = "utf-8") -> None:
    """Atomically write text content to a target file via a temporary file in the same directory."""
    path = Path(target_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write to a temp file in the same parent directory to ensure atomic os.replace across filesystem boundaries
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding=encoding) as tmp_file:
        tmp_file.write(content)
        tmp_file.flush()
        os.fsync(tmp_file.fileno())
        temp_name = tmp_file.name

    os.replace(temp_name, path)


def atomic_write_json(target_path: Union[str, Path], data: Any, indent: int = 2) -> None:
    """Atomically serialize data as formatted JSON to the target path."""
    content = json.dumps(data, indent=indent, ensure_ascii=False)
    atomic_write_file(target_path, content)


def atomic_write_yaml(target_path: Union[str, Path], data: Any) -> None:
    """Atomically serialize data as YAML to the target path."""
    content = yaml.dump(data, sort_keys=False)
    atomic_write_file(target_path, content)


def save_run(
    run_dir: Union[str, Path],
    envelope: ResultsEnvelope,
    metrics: StructuredMetrics,
    config_snapshot: Dict[str, Any],
) -> Path:
    """Persist run results atomically in the structured directory layout."""
    dir_path = Path(run_dir).resolve()
    dir_path.mkdir(parents=True, exist_ok=True)

    # 1. results.json
    atomic_write_file(
        dir_path / "results.json",
        envelope.model_dump_json(indent=2),
    )

    # 2. metrics.json
    atomic_write_file(
        dir_path / "metrics.json",
        metrics.model_dump_json(indent=2),
    )

    # 3. config.yaml
    atomic_write_yaml(
        dir_path / "config.yaml",
        config_snapshot,
    )

    return dir_path


def load_run(run_dir: Union[str, Path]) -> LoadedRun:
    """Load and validate an existing experiment run directory.

    Args:
        run_dir: Path to directory containing results.json, metrics.json, and config.yaml.

    Returns:
        LoadedRun instance.

    Raises:
        RunLoadError: If directory or required files are missing, or JSON is corrupt.
        UnsupportedSchemaVersionError: If schema_version in results.json is incompatible.
    """
    path = Path(run_dir).resolve()
    if not path.exists():
        raise RunLoadError(f"Run directory not found: '{path}'")
    if not path.is_dir():
        raise RunLoadError(f"Path is not a directory: '{path}'")

    results_file = path / "results.json"
    metrics_file = path / "metrics.json"
    config_file = path / "config.yaml"

    if not results_file.exists():
        raise RunLoadError(f"Missing required file 'results.json' in run directory '{path}'")
    if not metrics_file.exists():
        raise RunLoadError(f"Missing required file 'metrics.json' in run directory '{path}'")

    # Read and parse results.json
    try:
        results_raw = json.loads(results_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise RunLoadError(f"Corrupt or invalid JSON in '{results_file}': {e}") from e
    except Exception as e:
        raise RunLoadError(f"Failed to read '{results_file}': {e}") from e

    schema_version = results_raw.get("schema_version")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise UnsupportedSchemaVersionError(
            f"Unsupported schema_version '{schema_version}' in '{results_file}'. "
            f"Supported versions: {sorted(list(SUPPORTED_SCHEMA_VERSIONS))}"
        )

    try:
        envelope = ResultsEnvelope.model_validate(results_raw)
    except ValidationError as e:
        raise RunLoadError(f"Schema validation failed for '{results_file}':\n{e}") from e

    # Read and parse metrics.json
    try:
        metrics_raw = json.loads(metrics_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise RunLoadError(f"Corrupt or invalid JSON in '{metrics_file}': {e}") from e
    except Exception as e:
        raise RunLoadError(f"Failed to read '{metrics_file}': {e}") from e

    try:
        metrics = StructuredMetrics.model_validate(metrics_raw)
    except ValidationError as e:
        raise RunLoadError(f"Schema validation failed for '{metrics_file}':\n{e}") from e

    # Read config.yaml if present
    config_snapshot: Dict[str, Any] = {}
    if config_file.exists():
        try:
            config_snapshot = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
        except Exception:
            config_snapshot = {}

    return LoadedRun(
        envelope=envelope,
        metrics=metrics,
        config_snapshot=config_snapshot,
        run_dir=str(path),
    )


def list_saved_runs(reports_dir: Union[str, Path] = "reports") -> List[LoadedRun]:
    """List all saved experiment runs in reports_dir, ordered newest first."""
    base_dir = Path(reports_dir).resolve()
    if not base_dir.exists() or not base_dir.is_dir():
        return []

    runs: List[LoadedRun] = []
    # Search for directories that contain a results.json
    for child in base_dir.iterdir():
        if child.is_dir() and (child / "results.json").exists():
            try:
                run = load_run(child)
                runs.append(run)
            except Exception:
                # Skip invalid or partially written run folders during listing
                continue

    # Sort newest first by finished_at timestamp
    runs.sort(key=lambda r: r.envelope.finished_at, reverse=True)
    return runs
