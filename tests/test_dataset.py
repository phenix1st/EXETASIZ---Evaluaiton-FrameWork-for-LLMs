"""Tests for dataset loader handling valid and invalid datasets."""

import json
import pytest

from exetasiz.core.dataset import DatasetLoadError, load_dataset


def test_load_dataset_json_object(tmp_path):
    data = {
        "name": "sample",
        "test_cases": [
            {"id": "1", "category": "math", "prompt": "1+1", "expected_answer": "2"},
            {"id": "2", "category": "math", "prompt": "2+2", "expected_answer": "4"},
        ],
    }
    file = tmp_path / "data.json"
    file.write_text(json.dumps(data), encoding="utf-8")

    ds = load_dataset(file)
    assert ds.name == "sample"
    assert len(ds.test_cases) == 2


def test_load_dataset_json_list(tmp_path):
    data = [
        {"id": "1", "category": "math", "prompt": "1+1", "expected_answer": "2"},
    ]
    file = tmp_path / "list_data.json"
    file.write_text(json.dumps(data), encoding="utf-8")

    ds = load_dataset(file)
    assert ds.name == "list_data"
    assert len(ds.test_cases) == 1


def test_load_dataset_jsonl(tmp_path):
    lines = [
        json.dumps({"id": "1", "category": "cat1", "prompt": "p1", "expected_answer": "a1"}),
        json.dumps({"id": "2", "category": "cat2", "prompt": "p2", "expected_answer": "a2"}),
    ]
    file = tmp_path / "data.jsonl"
    file.write_text("\n".join(lines), encoding="utf-8")

    ds = load_dataset(file)
    assert ds.name == "data"
    assert len(ds.test_cases) == 2


def test_load_dataset_duplicate_ids(tmp_path):
    data = [
        {"id": "dup_1", "category": "cat1", "prompt": "p1", "expected_answer": "a1"},
        {"id": "dup_1", "category": "cat2", "prompt": "p2", "expected_answer": "a2"},
    ]
    file = tmp_path / "dups.json"
    file.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(DatasetLoadError, match="Duplicate test case IDs"):
        load_dataset(file)


def test_load_dataset_nonexistent():
    with pytest.raises(DatasetLoadError, match="not found"):
        load_dataset("nonexistent_file.json")


def test_load_dataset_empty_file(tmp_path):
    file = tmp_path / "empty.json"
    file.write_text("", encoding="utf-8")

    with pytest.raises(DatasetLoadError, match="is empty"):
        load_dataset(file)


def test_load_dataset_malformed_json(tmp_path):
    file = tmp_path / "bad.json"
    file.write_text("{invalid json", encoding="utf-8")

    with pytest.raises(DatasetLoadError, match="Malformed JSON"):
        load_dataset(file)


def test_load_dataset_invalid_schema(tmp_path):
    data = [{"id": "1", "prompt": "missing fields"}]
    file = tmp_path / "invalid_schema.json"
    file.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(DatasetLoadError, match="validation failed"):
        load_dataset(file)


def test_load_dataset_no_test_cases(tmp_path):
    data = {"name": "empty_set", "test_cases": []}
    file = tmp_path / "no_cases.json"
    file.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(DatasetLoadError, match="contains no test cases"):
        load_dataset(file)
