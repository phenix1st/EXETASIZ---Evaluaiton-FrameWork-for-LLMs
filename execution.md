# Exetasiz - Execution & Testing Guide

This guide covers how to run tests, execute experiments, and generate standalone HTML reports with Exetasiz.

---

## 1. Run the Complete Automated Test Suite (Pytest)

Run all **52 offline tests** covering schemas, dataset validation, mock providers, runner error resilience, pandas metrics, structured persistence, atomic writes, SVG charts, and CLI commands:

```bash
python -m pytest -v
```

### Run Specific Test Modules

```bash
# Persistence, atomic write & load_run tests
python -m pytest tests/test_persistence.py -v

# Standalone HTML report generation, SVG charts & XSS protection tests
python -m pytest tests/test_reporting.py -v

# CLI report command tests (report, report latest, list-runs)
python -m pytest tests/test_cli_reporting.py -v

# Core schema tests
python -m pytest tests/test_schemas.py -v

# Dataset ingestion & validation tests
python -m pytest tests/test_dataset.py -v

# Evaluator tests (ExactMatch, numeric-aware, stubs)
python -m pytest tests/test_evaluators.py -v

# Runner execution & error handling tests
python -m pytest tests/test_runner.py -v

# Metrics aggregation tests
python -m pytest tests/test_metrics.py -v

# Core CLI command tests
python -m pytest tests/test_cli.py -v
```

---

## 2. Execute Experiments via CLI

### A. Run Experiment with Auto-Generated HTML Report
Runs the demo dataset across `mock-perfect` (100% accuracy) and `mock-flawed` (50% accuracy) completely offline and generates `report.html` immediately upon completion:

```bash
python -m exetasiz.cli run experiments/basic_qa_demo.yaml --report
```

### B. Run Experiment with Category-Level Breakdown
Displays the breakdown of accuracy, pass rates, latencies, and token counts grouped by category (`reasoning`, `factual`, `classification`):

```bash
python -m exetasiz.cli run experiments/basic_qa_demo.yaml --by-category
```

### C. Standard Experiment Run (Structured Persistence Only)
Runs the experiment and saves `config.yaml`, `results.json`, and `metrics.json` to `reports/<experiment>_<timestamp>/`:

```bash
python -m exetasiz.cli run experiments/basic_qa_demo.yaml
```

---

## 3. Generate & View Standalone HTML Reports

### A. Open the Latest Report in Your Browser
Compiles the report for the newest experiment run in `reports/` and launches it in your default web browser:

```bash
python -m exetasiz.cli report latest --open
```

### B. Generate a Report for a Specific Run Directory
```bash
python -m exetasiz.cli report reports/<run_directory_name>
```

### C. Custom Destination and Report Title
```bash
python -m exetasiz.cli report latest --output my_benchmark_report.html --title "LLM Benchmark Sprint"
```

---

## 4. List Saved Runs

Display a table of all persisted evaluation runs sorted newest first:

```bash
python -m exetasiz.cli list-runs
```

Sample output:
```
run_directory                                   experiment     finished_at          models                     accuracy
----------------------------------------------  -------------  -------------------  -------------------------  ----------
basic_qa_demo_2026-10-07T22-21-44-612369+00-00  basic_qa_demo  2026-10-07 22:21:44  mock-perfect, mock-flawed  75.0%
```

---

## 5. Dataset & Provider Utilities

### Validate Datasets
Validates dataset formatting (JSON/JSONL) and asserts unique test case IDs:

```bash
python -m exetasiz.cli validate-dataset datasets/sample_qa.json
```

### List Registered Providers and Evaluators
```bash
python -m exetasiz.cli list-providers
python -m exetasiz.cli list-evaluators
```

---

## 6. Test Custom Evaluators & Extensibility

Run the custom evaluator demonstration script:

```bash
python examples/custom_evaluator_example.py
```

---

## 7. Inspect Saved Artifacts & Reports

Every run directory (`reports/<experiment>_<timestamp>/`) contains:
- **`config.yaml`**: Snapshot of the experiment configuration used.
- **`results.json`**: Envelope with `schema_version`, run metadata, and every individual `EvalResult`.
- **`metrics.json`**: Hierarchical metrics breakdown (`overall`, `by_model`, `by_category`, `by_model_and_category`).
- **`report.html`**: Fully standalone, single-file HTML report with inline CSS, SVG charts, and interactive filtering.

A pre-rendered demo report is also committed at:
```
examples/sample_report.html
```
