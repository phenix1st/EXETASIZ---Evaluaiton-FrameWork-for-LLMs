"""Pure-Python inline SVG chart generator for Exetasiz HTML reports.

Generates self-contained, light/dark-mode adaptable, accessible SVG charts.
No external charting libraries (e.g. matplotlib/plotly) used.
"""

from __future__ import annotations

import html
from typing import Dict, List, Optional
from exetasiz.core.models import AggregateMetric

# Curated categorical color palette for models/categories (works in light and dark mode)
COLOR_PALETTE = [
    "#4361ee",  # primary blue
    "#3a0ca3",  # deep indigo
    "#7209b7",  # purple
    "#f72585",  # vibrant pink
    "#4cc9f0",  # cyan
    "#06d6a0",  # emerald green
    "#ffd166",  # amber yellow
    "#ef476f",  # coral red
    "#118ab2",  # teal
    "#073b4c",  # midnight navy
]


def _escape(text: str) -> str:
    """Escape XML special characters."""
    return html.escape(str(text), quote=True)


def render_accuracy_chart(by_model: Dict[str, AggregateMetric]) -> str:
    """Render horizontal bar chart for model accuracy (0.0 to 1.0).

    Returns self-contained SVG string.
    """
    models = list(by_model.keys())
    if not models:
        return "<p class='empty-chart'>No model data available.</p>"

    bar_height = 28
    gap = 16
    left_margin = 160
    right_margin = 80
    top_margin = 35
    bottom_margin = 35
    chart_width = 720
    plot_width = chart_width - left_margin - right_margin
    total_height = top_margin + len(models) * (bar_height + gap) + bottom_margin

    svg = [
        f'<svg class="exetasiz-chart" width="100%" height="{total_height}" viewBox="0 0 {chart_width} {total_height}" '
        f'role="img" aria-label="Model Accuracy Comparison Chart" xmlns="http://www.w3.org/2000/svg">',
        '  <style>',
        '    .grid-line { stroke: var(--chart-grid, #e2e8f0); stroke-width: 1; stroke-dasharray: 3 3; }',
        '    .axis-label { fill: var(--chart-text, #64748b); font-size: 11px; font-family: inherit; }',
        '    .model-label { fill: var(--chart-text-strong, #1e293b); font-size: 12px; font-weight: 500; font-family: inherit; text-anchor: end; }',
        '    .val-label { fill: var(--chart-text-strong, #1e293b); font-size: 11px; font-weight: 600; font-family: inherit; }',
        '    .bar-rect { rx: 4; transition: opacity 0.2s; }',
        '    .bar-rect:hover { opacity: 0.85; }',
        '  </style>',
    ]

    # Grid vertical lines (0.0, 0.25, 0.5, 0.75, 1.0)
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x = left_margin + tick * plot_width
        svg.append(f'  <line x1="{x:.1f}" y1="{top_margin}" x2="{x:.1f}" y2="{total_height - bottom_margin}" class="grid-line" />')
        svg.append(f'  <text x="{x:.1f}" y="{total_height - bottom_margin + 16}" text-anchor="middle" class="axis-label">{int(tick * 100)}%</text>')

    # Bars
    for idx, model in enumerate(models):
        y = top_margin + idx * (bar_height + gap)
        metric = by_model[model]
        acc = max(0.0, min(1.0, metric.accuracy))
        width = acc * plot_width
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]

        # Model label (truncated if too long)
        clean_model = model if len(model) <= 22 else model[:20] + "…"
        svg.append(f'  <text x="{left_margin - 12}" y="{y + bar_height / 2 + 4:.1f}" class="model-label">{_escape(clean_model)}</text>')

        # Background track
        svg.append(f'  <rect x="{left_margin}" y="{y}" width="{plot_width}" height="{bar_height}" fill="var(--chart-bg-bar, #f1f5f9)" rx="4" />')
        # Active bar
        if width > 0:
            svg.append(f'  <rect x="{left_margin}" y="{y}" width="{width:.1f}" height="{bar_height}" fill="{color}" class="bar-rect" />')

        # Value label
        val_text = f"{metric.accuracy * 100:.1f}%"
        svg.append(f'  <text x="{left_margin + width + 8:.1f}" y="{y + bar_height / 2 + 4:.1f}" class="val-label">{val_text}</text>')

    svg.append('</svg>')
    return "\n".join(svg)


def render_latency_chart(by_model: Dict[str, AggregateMetric]) -> str:
    """Render bar chart comparing average and p95 latency per model in seconds."""
    models = list(by_model.keys())
    if not models:
        return "<p class='empty-chart'>No latency data available.</p>"

    # Find max latency across all models for scaling
    max_lat = 0.001
    for m in models:
        metric = by_model[m]
        p95 = metric.p95_latency_s or 0.0
        avg = metric.avg_latency_s or 0.0
        max_lat = max(max_lat, p95, avg)

    # Pad max_lat by 15%
    max_lat *= 1.15

    bar_group_height = 42
    gap = 20
    left_margin = 160
    right_margin = 80
    top_margin = 40
    bottom_margin = 40
    chart_width = 720
    plot_width = chart_width - left_margin - right_margin
    total_height = top_margin + len(models) * (bar_group_height + gap) + bottom_margin

    single_bar_h = 16

    svg = [
        f'<svg class="exetasiz-chart" width="100%" height="{total_height}" viewBox="0 0 {chart_width} {total_height}" '
        f'role="img" aria-label="Model Latency Comparison Chart" xmlns="http://www.w3.org/2000/svg">',
        '  <style>',
        '    .grid-line { stroke: var(--chart-grid, #e2e8f0); stroke-width: 1; stroke-dasharray: 3 3; }',
        '    .axis-label { fill: var(--chart-text, #64748b); font-size: 11px; font-family: inherit; }',
        '    .model-label { fill: var(--chart-text-strong, #1e293b); font-size: 12px; font-weight: 500; font-family: inherit; text-anchor: end; }',
        '    .val-label { fill: var(--chart-text-strong, #1e293b); font-size: 10px; font-family: inherit; }',
        '    .legend-text { fill: var(--chart-text, #64748b); font-size: 11px; font-family: inherit; }',
        '  </style>',
        # Legend at top
        '  <g transform="translate(160, 15)">',
        '    <rect x="0" y="0" width="12" height="12" fill="#4361ee" rx="2" />',
        '    <text x="18" y="10" class="legend-text">Avg Latency (s)</text>',
        '    <rect x="130" y="0" width="12" height="12" fill="#f72585" rx="2" />',
        '    <text x="148" y="10" class="legend-text">p95 Latency (s)</text>',
        '  </g>',
    ]

    # Grid ticks (0, 0.5 max, max)
    ticks = [0.0, max_lat * 0.5, max_lat]
    for tick in ticks:
        x = left_margin + (tick / max_lat) * plot_width
        svg.append(f'  <line x1="{x:.1f}" y1="{top_margin}" x2="{x:.1f}" y2="{total_height - bottom_margin}" class="grid-line" />')
        svg.append(f'  <text x="{x:.1f}" y="{total_height - bottom_margin + 16}" text-anchor="middle" class="axis-label">{tick:.3f}s</text>')

    for idx, model in enumerate(models):
        y = top_margin + idx * (bar_group_height + gap)
        metric = by_model[model]
        avg = metric.avg_latency_s or 0.0
        p95 = metric.p95_latency_s or 0.0

        w_avg = (avg / max_lat) * plot_width
        w_p95 = (p95 / max_lat) * plot_width

        clean_model = model if len(model) <= 22 else model[:20] + "…"
        svg.append(f'  <text x="{left_margin - 12}" y="{y + bar_group_height / 2 + 4:.1f}" class="model-label">{_escape(clean_model)}</text>')

        # Avg bar
        svg.append(f'  <rect x="{left_margin}" y="{y}" width="{max(1.0, w_avg):.1f}" height="{single_bar_h}" fill="#4361ee" rx="3" />')
        svg.append(f'  <text x="{left_margin + w_avg + 6:.1f}" y="{y + single_bar_h - 3:.1f}" class="val-label">{avg:.3f}s</text>')

        # p95 bar
        svg.append(f'  <rect x="{left_margin}" y="{y + single_bar_h + 4}" width="{max(1.0, w_p95):.1f}" height="{single_bar_h}" fill="#f72585" rx="3" />')
        svg.append(f'  <text x="{left_margin + w_p95 + 6:.1f}" y="{y + single_bar_h + 4 + single_bar_h - 3:.1f}" class="val-label">{p95:.3f}s</text>')

    svg.append('</svg>')
    return "\n".join(svg)


def render_grouped_category_chart(
    by_model_and_category: Dict[str, Dict[str, AggregateMetric]],
    categories: List[str],
) -> str:
    """Render grouped bar chart showing accuracy for each category grouped by model."""
    models = list(by_model_and_category.keys())
    if not models or not categories:
        return "<p class='empty-chart'>No category comparison data available.</p>"

    chart_width = 720
    top_margin = 60
    bottom_margin = 60
    left_margin = 60
    right_margin = 40
    plot_width = chart_width - left_margin - right_margin
    plot_height = 240
    total_height = top_margin + plot_height + bottom_margin

    cat_count = len(categories)
    group_width = plot_width / max(1, cat_count)
    bar_width = min(28, (group_width * 0.75) / max(1, len(models)))
    inner_gap = 4

    svg = [
        f'<svg class="exetasiz-chart" width="100%" height="{total_height}" viewBox="0 0 {chart_width} {total_height}" '
        f'role="img" aria-label="Category Accuracy Grouped Bar Chart" xmlns="http://www.w3.org/2000/svg">',
        '  <style>',
        '    .grid-line { stroke: var(--chart-grid, #e2e8f0); stroke-width: 1; stroke-dasharray: 3 3; }',
        '    .axis-label { fill: var(--chart-text, #64748b); font-size: 11px; font-family: inherit; }',
        '    .category-label { fill: var(--chart-text-strong, #1e293b); font-size: 12px; font-weight: 600; font-family: inherit; text-anchor: middle; }',
        '    .legend-text { fill: var(--chart-text, #64748b); font-size: 11px; font-family: inherit; }',
        '    .cat-bar { rx: 3; }',
        '  </style>',
    ]

    # Legend at top
    legend_items = []
    x_leg = left_margin
    y_leg = 20
    for idx, model in enumerate(models):
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        clean_model = model if len(model) <= 18 else model[:16] + "…"
        legend_items.append(
            f'  <g transform="translate({x_leg}, {y_leg})">'
            f'    <rect x="0" y="0" width="10" height="10" fill="{color}" rx="2" />'
            f'    <text x="14" y="9" class="legend-text">{_escape(clean_model)}</text>'
            f'  </g>'
        )
        x_leg += min(180, len(clean_model) * 8 + 35)
        if x_leg > chart_width - 120 and idx < len(models) - 1:
            x_leg = left_margin
            y_leg += 18
    svg.extend(legend_items)

    # Horizontal grid lines (0%, 25%, 50%, 75%, 100%)
    for pct in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = top_margin + plot_height - (pct * plot_height)
        svg.append(f'  <line x1="{left_margin}" y1="{y:.1f}" x2="{left_margin + plot_width}" y2="{y:.1f}" class="grid-line" />')
        svg.append(f'  <text x="{left_margin - 8}" y="{y + 4:.1f}" text-anchor="end" class="axis-label">{int(pct * 100)}%</text>')

    # Grouped bars
    for c_idx, cat in enumerate(categories):
        center_x = left_margin + (c_idx * group_width) + (group_width / 2)
        total_bars_w = len(models) * bar_width + (len(models) - 1) * inner_gap
        start_x = center_x - (total_bars_w / 2)

        for m_idx, model in enumerate(models):
            bx = start_x + m_idx * (bar_width + inner_gap)
            metric = by_model_and_category.get(model, {}).get(cat)
            acc = metric.accuracy if metric else 0.0
            bh = max(2.0, acc * plot_height)
            by = top_margin + plot_height - bh
            color = COLOR_PALETTE[m_idx % len(COLOR_PALETTE)]

            svg.append(f'  <rect x="{bx:.1f}" y="{by:.1f}" width="{bar_width:.1f}" height="{bh:.1f}" fill="{color}" class="cat-bar">'
                       f'<title>{_escape(model)} - {_escape(cat)}: {acc*100:.1f}%</title></rect>')

        # Category label under group
        svg.append(f'  <text x="{center_x:.1f}" y="{top_margin + plot_height + 22}" class="category-label">{_escape(cat)}</text>')

    svg.append('</svg>')
    return "\n".join(svg)
