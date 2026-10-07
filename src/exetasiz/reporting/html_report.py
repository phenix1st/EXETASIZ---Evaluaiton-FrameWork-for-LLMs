"""Standalone HTML report generator for Exetasiz evaluation runs."""

from __future__ import annotations

import importlib.resources
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import jinja2
import yaml

from exetasiz.core.models import LoadedRun, TestCase
from exetasiz.core.persistence import atomic_write_file
from exetasiz.core.dataset import load_dataset
from exetasiz.reporting.charts import (
    render_accuracy_chart,
    render_grouped_category_chart,
    render_latency_chart,
)


def _load_asset(filename: str) -> str:
    """Load text template or asset from exetasiz.reporting.templates package safely."""
    try:
        # Python 3.9+ standard importlib.resources.files
        files = importlib.resources.files("exetasiz.reporting.templates")
        asset_file = files.joinpath(filename)
        return asset_file.read_text(encoding="utf-8")
    except Exception:
        # Fallback to local relative path if package not installed as site-package yet
        template_dir = Path(__file__).resolve().parent / "templates"
        return (template_dir / filename).read_text(encoding="utf-8")


def generate_html_report(
    run: LoadedRun,
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
) -> Path:
    """Generates a standalone, single-file HTML evaluation report.

    Args:
        run: LoadedRun instance with parsed results envelope, metrics, and config.
        output_path: Optional destination path. Defaults to <run_dir>/report.html.
        title: Optional report document title.

    Returns:
        Path to the generated report file.
    """
    out_file = (
        Path(output_path).resolve()
        if output_path is not None
        else Path(run.run_dir) / "report.html"
    )

    report_title = title or f"Exetasiz Report - {run.envelope.experiment_name}"

    # Inlined CSS
    inlined_css = _load_asset("styles.css")
    template_src = _load_asset("report.html.j2")

    # Safe Jinja environment with autoescape enabled
    env = jinja2.Environment(
        autoescape=jinja2.select_autoescape(
            enabled_extensions=["html", "xml", "j2"],
            default_for_string=True,
        )
    )
    template = env.from_string(template_src)

    # Categories list
    categories = sorted(list(run.metrics.by_category.keys()))

    # Attempt to load original dataset to supply prompts and expected answers
    test_cases_map: Dict[str, TestCase] = {}
    ds_path_str = run.envelope.dataset.path
    if ds_path_str:
        candidate_paths = [
            Path(ds_path_str),
            Path(run.run_dir) / ds_path_str,
            Path(run.run_dir).parent.parent / ds_path_str,
        ]
        for p in candidate_paths:
            if p.exists() and p.is_file():
                try:
                    ds = load_dataset(p)
                    for tc in ds.test_cases:
                        test_cases_map[tc.id] = tc
                    break
                except Exception:
                    pass

    # Model comparisons highlights
    by_model = run.metrics.by_model
    best_model: Optional[str] = None
    best_acc: Optional[float] = None
    fastest_model: Optional[str] = None
    lowest_lat: Optional[float] = None

    if by_model:
        # Highest accuracy
        sorted_by_acc = sorted(by_model.items(), key=lambda kv: kv[1].accuracy, reverse=True)
        best_model = sorted_by_acc[0][0]
        best_acc = sorted_by_acc[0][1].accuracy

        # Lowest avg latency (among non-null and > 0)
        models_with_lat = [
            (m, metric)
            for m, metric in by_model.items()
            if metric.avg_latency_s is not None and metric.avg_latency_s > 0
        ]
        if models_with_lat:
            sorted_by_lat = sorted(models_with_lat, key=lambda kv: kv[1].avg_latency_s)  # type: ignore
            fastest_model = sorted_by_lat[0][0]
            lowest_lat = sorted_by_lat[0][1].avg_latency_s

    # Generate pure inline SVG charts
    chart_accuracy = render_accuracy_chart(by_model)
    chart_latency = render_latency_chart(by_model)
    chart_categories = render_grouped_category_chart(run.metrics.by_model_and_category, categories)

    # Render context
    context = {
        "title": report_title,
        "run": run,
        "inlined_css": inlined_css,
        "categories": categories,
        "test_cases_map": test_cases_map,
        "best_model": best_model,
        "best_accuracy": best_acc,
        "fastest_model": fastest_model,
        "lowest_latency": lowest_lat,
        "chart_accuracy": chart_accuracy,
        "chart_latency": chart_latency,
        "chart_categories": chart_categories,
        "config_yaml_str": yaml.dump(run.config_snapshot, sort_keys=False) if run.config_snapshot else "# None",
        "generation_time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }

    rendered_html = template.render(**context)
    atomic_write_file(out_file, rendered_html, encoding="utf-8")
    return out_file
