"""Dataset loading and validation functionality."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union
from pydantic import ValidationError

from exetasiz.core.models import Dataset, TestCase


class DatasetLoadError(Exception):
    """Raised when a dataset cannot be read or validated."""


def load_dataset(file_path: Union[str, Path]) -> Dataset:
    """Load and validate a dataset from a JSON or JSONL file.

    Args:
        file_path: Path to the dataset file.

    Returns:
        Dataset: Validated Dataset instance containing test cases.

    Raises:
        DatasetLoadError: If file is not found, malformed, contains duplicate IDs,
                          or fails schema validation.
    """
    path = Path(file_path)
    if not path.exists():
        raise DatasetLoadError(f"Dataset file not found: {path}")

    suffix = path.suffix.lower()
    raw_data: Any

    try:
        content = path.read_text(encoding="utf-8").strip()
    except Exception as e:
        raise DatasetLoadError(f"Failed to read dataset file '{path}': {e}") from e

    if not content:
        raise DatasetLoadError(f"Dataset file '{path}' is empty.")

    if suffix == ".jsonl":
        test_cases_raw: List[Dict[str, Any]] = []
        for line_num, line in enumerate(content.splitlines(), start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                item = json.loads(line_str)
                test_cases_raw.append(item)
            except json.JSONDecodeError as e:
                raise DatasetLoadError(
                    f"Malformed JSON on line {line_num} in '{path}': {e}"
                ) from e
        dataset_name = path.stem
        raw_data = {"name": dataset_name, "test_cases": test_cases_raw}
    elif suffix == ".json":
        try:
            raw_data = json.loads(content)
        except json.JSONDecodeError as e:
            raise DatasetLoadError(f"Malformed JSON in '{path}': {e}") from e

        if isinstance(raw_data, list):
            raw_data = {"name": path.stem, "test_cases": raw_data}
        elif isinstance(raw_data, dict):
            if "test_cases" not in raw_data:
                raise DatasetLoadError(
                    f"JSON object in '{path}' must contain a 'test_cases' list."
                )
            if "name" not in raw_data:
                raw_data["name"] = path.stem
        else:
            raise DatasetLoadError(
                f"Expected JSON root to be an object or a list in '{path}'."
            )
    else:
        raise DatasetLoadError(
            f"Unsupported file format '{suffix}'. Exetasiz supports .json and .jsonl files."
        )

    # Validate schema
    try:
        dataset = Dataset.model_validate(raw_data)
    except ValidationError as e:
        raise DatasetLoadError(f"Dataset schema validation failed for '{path}':\n{e}") from e

    if len(dataset.test_cases) == 0:
        raise DatasetLoadError(f"Dataset '{dataset.name}' in '{path}' contains no test cases.")

    # Check for duplicate IDs
    seen_ids = set()
    duplicates = []
    for tc in dataset.test_cases:
        if tc.id in seen_ids:
            duplicates.append(tc.id)
        seen_ids.add(tc.id)

    if duplicates:
        unique_dupes = sorted(list(set(duplicates)))
        raise DatasetLoadError(
            f"Duplicate test case IDs found in dataset '{dataset.name}': {unique_dupes}"
        )

    return dataset
