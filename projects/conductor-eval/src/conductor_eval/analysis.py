import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from conductor_core.music import (
    INTERVAL_NAMES,
    NOTE_NAMES,
    beats_to_duration_name,
    note_name_to_pitch_class,
    pitch_class_to_interval,
    pitch_class_to_note,
)
from dash import Input, Output, dcc, html

PLOTLY_BG = "#1a1a2e"
PLOTLY_GRID = "#2a2a4a"
PLOTLY_TEXT = "#e0e0e0"
PLOTLY_ACCENT = "#0f3460"
PLOTLY_CARD_BG = "#16213e"


def apply_plotly_theme(fig):
    """Apply the Conductor Eval dashboard theme to a Plotly figure."""
    fig.update_layout(
        paper_bgcolor=PLOTLY_BG,
        plot_bgcolor=PLOTLY_BG,
        font=dict(color=PLOTLY_TEXT, family="Segoe UI, sans-serif"),
        xaxis=dict(gridcolor=PLOTLY_GRID, zerolinecolor=PLOTLY_GRID),
        yaxis=dict(gridcolor=PLOTLY_GRID, zerolinecolor=PLOTLY_GRID),
        margin=dict(l=60, r=30, t=50, b=60),
    )
    return fig


logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_COLORS = px.colors.qualitative.Set2
DEFAULT_EVALUATIONS_DIR = Path("evaluations")

# Plotly textposition options mapped to angles (degrees, counter-clockwise from +x axis).
# The label is placed in the direction of the angle relative to the marker.
_TEXT_POSITIONS = [
    (0, "middle right"),
    (45, "top right"),
    (90, "top center"),
    (135, "top left"),
    (180, "middle left"),
    (225, "bottom left"),
    (270, "bottom center"),
    (315, "bottom right"),
]


def compute_text_positions(x_vals, y_vals):
    """Compute per-point textposition strings to minimize label overlap on scatter plots.

    Uses a repulsion-based approach: for each point, computes a net repulsion vector
    from all other points (weighted by inverse squared distance in normalized space),
    then places the label in the direction that points away from the densest region.

    Args:
        x_vals (list | pd.Series): X coordinates of the scatter points.
        y_vals (list | pd.Series): Y coordinates of the scatter points.

    Returns:
        list[str]: A list of Plotly textposition strings, one per point.
    """
    import math

    xs = list(x_vals)
    ys = list(y_vals)
    n = len(xs)

    if n <= 1:
        return ["top center"] * n

    # Normalize to [0, 1] so x and y distances are comparable
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_range = x_max - x_min if x_max != x_min else 1.0
    y_range = y_max - y_min if y_max != y_min else 1.0
    xn = [(v - x_min) / x_range for v in xs]
    yn = [(v - y_min) / y_range for v in ys]

    positions = []
    for i in range(n):
        # Sum repulsion vectors from all other points
        rx, ry = 0.0, 0.0
        for j in range(n):
            if i == j:
                continue
            dx = xn[i] - xn[j]
            dy = yn[i] - yn[j]
            dist_sq = dx * dx + dy * dy
            if dist_sq < 1e-12:
                # Near-identical points: push in arbitrary direction
                rx += 0.0
                ry += 1.0
                continue
            # Inverse-square weighting so nearby points dominate
            weight = 1.0 / dist_sq
            rx += dx * weight
            ry += dy * weight

        if abs(rx) < 1e-9 and abs(ry) < 1e-9:
            positions.append("top center")
            continue

        angle = math.degrees(math.atan2(ry, rx)) % 360

        # Find the closest named position
        best_pos = "top center"
        best_diff = 360.0
        for ref_angle, pos_name in _TEXT_POSITIONS:
            diff = abs(angle - ref_angle)
            if diff > 180:
                diff = 360 - diff
            if diff < best_diff:
                best_diff = diff
                best_pos = pos_name
        positions.append(best_pos)

    return positions


def load_run(run_path):
    """Load all evaluation data from a run directory into a DataFrame.

    Recursively walks the results/ tree, reads every test_results.json,
    and flattens each into a single row.

    Args:
        run_path (str): Path to the run directory.

    Returns:
        tuple: (pd.DataFrame of all results, dict of config, dict of summary)
    """
    run_path = Path(run_path)

    # Load config
    config_path = run_path / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Load summary
    summary_path = run_path / "summary.json"
    summary = {}
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

    # Collect all test_results.json files
    rows = []
    results_dir = run_path / "results"
    if not results_dir.exists():
        logger.warning("No results directory found in %s", run_path)
        return pd.DataFrame(), config, summary

    for tr_path in results_dir.rglob("test_results.json"):
        with open(tr_path, "r", encoding="utf-8") as f:
            result = json.load(f)

        # Determine variation from directory structure
        rel = tr_path.relative_to(results_dir)
        parts = (
            rel.parts
        )  # e.g. ("Ollama", "model", "prompt_slug", "C_major", "low", "test_results.json")
        # If there's a subfolder beyond root_scale (5+ parts before the filename), it's a variation
        if len(parts) > 5:
            variation = parts[-2]
        else:
            variation = "standard"

        tests = result.get("tests", {})
        cfg = result.get("config", {})
        metrics = result.get("metrics", {})

        # Scale test details
        scale_test = tests.get("scale", {})
        duration_test = tests.get("duration", {})

        row = {
            "model": result.get("model", "unknown"),
            "provider": result.get("provider", "unknown"),
            "prompt": result.get("prompt", ""),
            "original_prompt": result.get("original_prompt", ""),
            "root": result.get("root", ""),
            "scale": result.get("scale", ""),
            "variation": variation,
            # Config
            "use_thinking": cfg.get("use_thinking", False),
            "effort": cfg.get("effort"),
            "temperature": cfg.get("temperature", 0.0),
            # Metrics
            "api_latency": metrics.get("api_latency", 0.0),
            "cost": metrics.get("cost", 0.0),
            # Overall
            "overall_pass": tests.get("overall_pass", False),
            "error": result.get("error"),
            # Scale test
            "scale_ran": scale_test.get("ran", False),
            "scale_total": scale_test.get("total", 0),
            "scale_correct": scale_test.get("correct", 0),
            "scale_incorrect": scale_test.get("incorrect", 0),
            "scale_pitches_correct": scale_test.get("pitches", {}).get("correct", []),
            "scale_pitches_incorrect": scale_test.get("pitches", {}).get("incorrect", []),
            # Duration test
            "duration_ran": duration_test.get("ran", False),
            "duration_total": duration_test.get("total", 0),
            "duration_correct": duration_test.get("correct", 0),
            "duration_incorrect": duration_test.get("incorrect", 0),
            "duration_lengths": duration_test.get("lengths", {}),
            "duration_param": duration_test.get("params", {}).get("duration", ""),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        # Compute derived columns
        df["has_error"] = df["error"].apply(lambda x: isinstance(x, str) and len(x) > 0)
        df["scale_accuracy"] = df.apply(
            lambda r: r["scale_correct"] / r["scale_total"] if r["scale_total"] > 0 else None,
            axis=1,
        )
        df["duration_accuracy"] = df.apply(
            lambda r: (
                r["duration_correct"] / r["duration_total"] if r["duration_total"] > 0 else None
            ),
            axis=1,
        )
        df["scale_pass"] = df.apply(lambda r: r["scale_incorrect"] == 0 and r["scale_ran"], axis=1)
        df["duration_pass"] = df.apply(
            lambda r: r["duration_incorrect"] == 0 and r["duration_ran"], axis=1
        )

        # Build model instance names when reasoning is being tested.
        # This overwrites the "model" column so every existing chart function
        # automatically renders model+effort instances with zero code changes.
        df["base_model"] = df["model"].copy()
        if config.get("test_reasoning", False):

            def _instance_name(row):
                if row["effort"] is not None and pd.notna(row["effort"]):
                    return f"{row['base_model']} ({row['effort']})"
                elif row["use_thinking"]:
                    return f"{row['base_model']} (reasoning)"
                return row["base_model"]

            df["model"] = df.apply(_instance_name, axis=1)

            # For legacy toggle models (no effort, but both thinking=True and
            # thinking=False rows exist for the same base model), tag the
            # non-thinking rows as "(standard)" so they are distinguishable.
            for base in df["base_model"].unique():
                mask = df["base_model"] == base
                subset = df.loc[mask]
                no_effort = subset["effort"].isna().all()
                has_thinking = subset["use_thinking"].any()
                has_standard = not subset["use_thinking"].all()
                if no_effort and has_thinking and has_standard:
                    std_mask = mask & ~df["use_thinking"]
                    df.loc[std_mask, "model"] = df.loc[std_mask, "base_model"] + " (standard)"

    logger.info("Loaded %d results from %s", len(df), run_path)
    return df, config, summary


def list_available_runs(base_dir=DEFAULT_EVALUATIONS_DIR):
    """List available evaluation runs in the base directory.

    Args:
        base_dir (str): Base directory containing run subdirectories.

    Returns:
        list[Path]: Sorted list of run directory paths.
    """
    base = Path(base_dir)
    if not base.exists():
        return []
    runs = [d for d in sorted(base.iterdir()) if d.is_dir() and (d / "config.json").exists()]
    return runs


def select_run_interactive(base_dir=DEFAULT_EVALUATIONS_DIR):
    """Prompt user to select a run from available runs.

    Args:
        base_dir (str): Base directory containing run subdirectories.

    Returns:
        Path: Selected run directory path.
    """
    runs = list_available_runs(base_dir)
    if not runs:
        print(f"No evaluation runs found in '{base_dir}/'")
        sys.exit(1)

    print("\nAvailable evaluation runs:")
    print("-" * 60)
    for i, run in enumerate(runs, 1):
        # Try to load config for a nice display
        try:
            with open(run / "config.json", "r") as f:
                cfg = json.load(f)
            name = cfg.get("run_name", run.name)
            ts = cfg.get("timestamp", "")
            models = [m[1] if isinstance(m, list) else m for m in cfg.get("models", [])]
            model_count = len(models)
            print(f"  [{i}] {name}  ({ts}, {model_count} models)")
        except Exception:
            print(f"  [{i}] {run.name}")
    print("-" * 60)

    while True:
        try:
            choice = input(f"Select a run (1-{len(runs)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(runs):
                return runs[idx]
            print(f"Please enter a number between 1 and {len(runs)}")
        except (ValueError, EOFError):
            print("Invalid input. Please enter a number.")


def apply_filters(df, models, roots, scales, variations):
    """Apply global filter selections to the DataFrame.

    Args:
        df (pd.DataFrame): Full results DataFrame.
        models (list): Selected model names.
        roots (list): Selected root notes.
        scales (list): Selected scale types.
        variations (list): Selected variation types.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    filtered = df.copy()
    if models:
        filtered = filtered[filtered["model"].isin(models)]
    if roots:
        filtered = filtered[filtered["root"].isin(roots)]
    if scales:
        filtered = filtered[filtered["scale"].isin(scales)]
    if variations:
        filtered = filtered[filtered["variation"].isin(variations)]
    return filtered


def build_pass_rate_by_model(df):
    """Build horizontal bar chart of overall pass rate by model.

    Args:
        df (pd.DataFrame): Filtered results DataFrame.

    Returns:
        go.Figure: Bar chart figure.
    """
    if df.empty:
        return apply_plotly_theme(go.Figure().update_layout(title="No data"))

    stats = (
        df.groupby("model")
        .agg(
            tested=("overall_pass", "count"),
            passed=("overall_pass", "sum"),
        )
        .reset_index()
    )
    stats["pass_rate"] = (stats["passed"] / stats["tested"] * 100).round(1)
    stats = stats.sort_values("pass_rate", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=stats["pass_rate"],
            y=stats["model"],
            orientation="h",
            text=[
                f"{r}% ({p}/{t})"
                for r, p, t in zip(
                    stats["pass_rate"],
                    stats["passed"].astype(int),
                    stats["tested"].astype(int),
                )
            ],
            textposition="auto",
            marker_color=[MODEL_COLORS[i % len(MODEL_COLORS)] for i in range(len(stats))],
        )
    )
    fig.update_layout(
        title="Overall Pass Rate by Model",
        xaxis_title="Pass Rate (%)",
        yaxis_title="",
        xaxis=dict(range=[0, 105]),
    )
    return apply_plotly_theme(fig)


def build_per_test_breakdown(df):
    """Build grouped bar chart showing scale vs duration pass rate per model.

    Args:
        df (pd.DataFrame): Filtered results DataFrame.

    Returns:
        go.Figure: Grouped bar chart figure.
    """
    if df.empty:
        return apply_plotly_theme(go.Figure().update_layout(title="No data"))

    # Only include rows where tests actually ran
    models = sorted(df["model"].unique())
    scale_rates = []
    duration_rates = []
    overall_rates = []

    for model in models:
        mdf = df[df["model"] == model]
        # Scale pass rate (only where scale test ran)
        scale_df = mdf[mdf["scale_ran"]]
        s_rate = (scale_df["scale_pass"].sum() / len(scale_df) * 100) if len(scale_df) > 0 else 0
        scale_rates.append(round(s_rate, 1))
        # Duration pass rate
        dur_df = mdf[mdf["duration_ran"]]
        d_rate = (dur_df["duration_pass"].sum() / len(dur_df) * 100) if len(dur_df) > 0 else 0
        duration_rates.append(round(d_rate, 1))
        # Overall
        o_rate = (mdf["overall_pass"].sum() / len(mdf) * 100) if len(mdf) > 0 else 0
        overall_rates.append(round(o_rate, 1))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Scale Test",
            x=models,
            y=scale_rates,
            text=scale_rates,
            textposition="auto",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Duration Test",
            x=models,
            y=duration_rates,
            text=duration_rates,
            textposition="auto",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Overall (Both)",
            x=models,
            y=overall_rates,
            text=overall_rates,
            textposition="auto",
        )
    )
    fig.update_layout(
        barmode="group",
        title="Pass Rate by Test Type per Model",
        xaxis_title="Model",
        yaxis_title="Pass Rate (%)",
        yaxis=dict(range=[0, 105]),
    )
    return apply_plotly_theme(fig)


def build_model_root_heatmap(df):
    """Build heatmap of pass rate at each (model, root) intersection.

    Args:
        df (pd.DataFrame): Filtered results DataFrame.

    Returns:
        go.Figure: Heatmap figure.
    """
    if df.empty:
        return apply_plotly_theme(go.Figure().update_layout(title="No data"))

    pivot = df.pivot_table(values="overall_pass", index="model", columns="root", aggfunc="mean")
    pivot = (pivot * 100).round(1)

    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            text=pivot.values,
            texttemplate="%{text:.1f}%",
            colorscale="RdYlGn",
            zmin=0,
            zmax=100,
            colorbar=dict(title="Pass %"),
        )
    )
    fig.update_layout(title="Pass Rate: Model x Root", xaxis_title="Root", yaxis_title="Model")
    return apply_plotly_theme(fig)


def build_model_scale_heatmap(df):
    """Build heatmap of pass rate at each (model, scale) intersection.

    Args:
        df (pd.DataFrame): Filtered results DataFrame.

    Returns:
        go.Figure: Heatmap figure.
    """
    if df.empty:
        return apply_plotly_theme(go.Figure().update_layout(title="No data"))

    pivot = df.pivot_table(values="overall_pass", index="model", columns="scale", aggfunc="mean")
    pivot = (pivot * 100).round(1)

    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            text=pivot.values,
            texttemplate="%{text:.1f}%",
            colorscale="RdYlGn",
            zmin=0,
            zmax=100,
            colorbar=dict(title="Pass %"),
        )
    )
    fig.update_layout(
        title="Pass Rate: Model x Scale", xaxis_title="Scale Type", yaxis_title="Model"
    )
    return apply_plotly_theme(fig)


def build_major_vs_minor_by_model(df):
    """Build grouped bar chart comparing major vs minor pass rates per model.

    Args:
        df (pd.DataFrame): Filtered results DataFrame.

    Returns:
        go.Figure: Grouped bar chart figure.
    """
    if df.empty:
        return apply_plotly_theme(go.Figure().update_layout(title="No data"))

    models = sorted(df["model"].unique())
    major_rates = []
    minor_rates = []

    for model in models:
        mdf = df[df["model"] == model]
        maj = mdf[mdf["scale"] == "major"]
        mn = mdf[mdf["scale"] == "minor"]
        major_rates.append(round(maj["overall_pass"].mean() * 100, 1) if len(maj) > 0 else 0)
        minor_rates.append(round(mn["overall_pass"].mean() * 100, 1) if len(mn) > 0 else 0)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Major",
            x=models,
            y=major_rates,
            text=major_rates,
            textposition="auto",
            marker_color="#5dade2",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Minor",
            x=models,
            y=minor_rates,
            text=minor_rates,
            textposition="auto",
            marker_color="#e74c3c",
        )
    )
    fig.update_layout(
        barmode="group",
        title="Major vs Minor Pass Rate by Model",
        xaxis_title="Model",
        yaxis_title="Pass Rate (%)",
        yaxis=dict(range=[0, 105]),
    )
    return apply_plotly_theme(fig)


def build_root_pass_rate(df):
    """Build bar chart of pass rate by root note.

    Args:
        df (pd.DataFrame): Filtered results DataFrame.

    Returns:
        go.Figure: Bar chart figure.
    """
    if df.empty:
        return apply_plotly_theme(go.Figure().update_layout(title="No data"))

    stats = (
        df.groupby("root")
        .agg(
            tested=("overall_pass", "count"),
            passed=("overall_pass", "sum"),
        )
        .reset_index()
    )
    stats["pass_rate"] = (stats["passed"] / stats["tested"] * 100).round(1)
    stats = stats.sort_values("pass_rate", ascending=False)

    fig = go.Figure(
        go.Bar(
            x=stats["root"],
            y=stats["pass_rate"],
            text=[
                f"{r}% ({p}/{t})"
                for r, p, t in zip(
                    stats["pass_rate"],
                    stats["passed"].astype(int),
                    stats["tested"].astype(int),
                )
            ],
            textposition="auto",
            marker_color="#5dade2",
        )
    )
    fig.update_layout(
        title="Pass Rate by Root Note",
        xaxis_title="Root Note",
        yaxis_title="Pass Rate (%)",
        yaxis=dict(range=[0, 105]),
    )
    return apply_plotly_theme(fig)


def build_root_scale_grouped(df):
    """Build grouped bar chart of major/minor pass rates per root note.

    Args:
        df (pd.DataFrame): Filtered results DataFrame.

    Returns:
        go.Figure: Grouped bar chart figure.
    """
    if df.empty:
        return apply_plotly_theme(go.Figure().update_layout(title="No data"))

    roots = sorted(df["root"].unique())
    major_rates = []
    minor_rates = []

    for root in roots:
        rdf = df[df["root"] == root]
        maj = rdf[rdf["scale"] == "major"]
        mn = rdf[rdf["scale"] == "minor"]
        major_rates.append(round(maj["overall_pass"].mean() * 100, 1) if len(maj) > 0 else 0)
        minor_rates.append(round(mn["overall_pass"].mean() * 100, 1) if len(mn) > 0 else 0)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Major",
            x=roots,
            y=major_rates,
            text=major_rates,
            textposition="auto",
            marker_color="#5dade2",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Minor",
            x=roots,
            y=minor_rates,
            text=minor_rates,
            textposition="auto",
            marker_color="#e74c3c",
        )
    )
    fig.update_layout(
        barmode="group",
        title="Pass Rate by Root Note (Major vs Minor)",
        xaxis_title="Root Note",
        yaxis_title="Pass Rate (%)",
        yaxis=dict(range=[0, 105]),
    )
    return apply_plotly_theme(fig)


def build_root_scale_heatmap(df):
    """Build heatmap of pass rate across all root x scale combinations.

    Args:
        df (pd.DataFrame): Filtered results DataFrame.

    Returns:
        go.Figure: Heatmap figure.
    """
    if df.empty:
        return apply_plotly_theme(go.Figure().update_layout(title="No data"))

    # Create combined label
    df_copy = df.copy()
    df_copy["root_scale"] = df_copy["root"] + " " + df_copy["scale"]

    pivot = df_copy.pivot_table(
        values="overall_pass", index="model", columns="root_scale", aggfunc="mean"
    )
    pivot = (pivot * 100).round(1)

    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            text=pivot.values,
            texttemplate="%{text:.1f}%",
            colorscale="RdYlGn",
            zmin=0,
            zmax=100,
            colorbar=dict(title="Pass %"),
        )
    )
    fig.update_layout(
        title="Pass Rate: Model x Root+Scale",
        xaxis_title="Root + Scale",
        yaxis_title="Model",
    )
    return apply_plotly_theme(fig)


def build_latency_box(df):
    """Build box plot of API latency per model, split by variation.

    Args:
        df (pd.DataFrame): Filtered results DataFrame.

    Returns:
        go.Figure: Box plot figure.
    """
    if df.empty:
        return apply_plotly_theme(go.Figure().update_layout(title="No data"))

    fig = go.Figure()
    models = sorted(df["model"].unique())

    for model in models:
        mdf = df[df["model"] == model]
        fig.add_trace(go.Box(y=mdf["api_latency"], name=model, boxmean=True))

    fig.update_layout(
        title="API Latency Distribution by Model",
        yaxis_title="Latency (seconds)",
        showlegend=False,
    )
    return apply_plotly_theme(fig)


def build_latency_vs_pass(df):
    """Build scatter plot of average latency vs pass rate per model.

    Args:
        df (pd.DataFrame): Filtered results DataFrame.

    Returns:
        go.Figure: Scatter plot figure.
    """
    if df.empty:
        return apply_plotly_theme(go.Figure().update_layout(title="No data"))

    stats = (
        df.groupby("model")
        .agg(
            avg_latency=("api_latency", "mean"),
            pass_rate=("overall_pass", "mean"),
            count=("overall_pass", "count"),
        )
        .reset_index()
    )
    stats["pass_rate"] = (stats["pass_rate"] * 100).round(1)

    text_positions = compute_text_positions(stats["avg_latency"], stats["pass_rate"])
    fig = go.Figure(
        go.Scatter(
            x=stats["avg_latency"],
            y=stats["pass_rate"],
            mode="markers+text",
            text=stats["model"],
            textposition=text_positions,
            marker=dict(
                size=stats["count"] / stats["count"].max() * 30 + 10,
                color=[MODEL_COLORS[i % len(MODEL_COLORS)] for i in range(len(stats))],
            ),
        )
    )
    fig.update_layout(
        title="Latency vs Pass Rate by Model",
        xaxis_title="Average Latency (seconds)",
        yaxis_title="Pass Rate (%)",
        yaxis=dict(range=[0, 105]),
    )
    return apply_plotly_theme(fig)


def build_cost_by_model(df):
    """Build bar chart of total cost by model.

    Args:
        df (pd.DataFrame): Filtered results DataFrame.

    Returns:
        go.Figure: Bar chart figure.
    """
    if df.empty:
        return apply_plotly_theme(go.Figure().update_layout(title="No data"))

    stats = (
        df.groupby("model")
        .agg(
            total_cost=("cost", "sum"),
            tested=("cost", "count"),
            passed=("overall_pass", "sum"),
        )
        .reset_index()
    )
    stats["cost_per_gen"] = (stats["total_cost"] / stats["tested"]).round(4)
    stats["cost_per_success"] = stats.apply(
        lambda r: round(r["total_cost"] / r["passed"], 4) if r["passed"] > 0 else 0,
        axis=1,
    )
    stats = stats.sort_values("total_cost", ascending=True)

    if stats["total_cost"].sum() == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="All costs are $0 (local models)",
            showarrow=False,
            font=dict(size=16, color=PLOTLY_TEXT),
        )
        fig.update_layout(title="Cost Analysis")
        return apply_plotly_theme(fig)

    fig = go.Figure(
        go.Bar(
            x=stats["total_cost"],
            y=stats["model"],
            orientation="h",
            text=[f"${c:.4f}" for c in stats["total_cost"]],
            textposition="auto",
            marker_color="#f39c12",
        )
    )
    fig.update_layout(
        title="Total Cost by Model",
        xaxis_title="Total Cost ($)",
        yaxis_title="",
    )
    return apply_plotly_theme(fig)


def build_cost_vs_pass(df):
    """Build scatter plot of cost per generation vs pass rate.

    Args:
        df (pd.DataFrame): Filtered results DataFrame.

    Returns:
        go.Figure: Scatter plot figure.
    """
    if df.empty:
        return apply_plotly_theme(go.Figure().update_layout(title="No data"))

    stats = (
        df.groupby("model")
        .agg(
            total_cost=("cost", "sum"),
            tested=("cost", "count"),
            pass_rate=("overall_pass", "mean"),
        )
        .reset_index()
    )

    if stats["total_cost"].sum() == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="All costs are $0 (local models)",
            showarrow=False,
            font=dict(size=16, color=PLOTLY_TEXT),
        )
        fig.update_layout(title="Cost vs Pass Rate")
        return apply_plotly_theme(fig)

    stats["cost_per_gen"] = stats["total_cost"] / stats["tested"]
    stats["pass_rate"] = (stats["pass_rate"] * 100).round(1)

    text_positions = compute_text_positions(stats["cost_per_gen"], stats["pass_rate"])
    fig = go.Figure(
        go.Scatter(
            x=stats["cost_per_gen"],
            y=stats["pass_rate"],
            mode="markers+text",
            text=stats["model"],
            textposition=text_positions,
            marker=dict(
                size=15,
                color=[MODEL_COLORS[i % len(MODEL_COLORS)] for i in range(len(stats))],
            ),
        )
    )
    fig.update_layout(
        title="Cost per Generation vs Pass Rate",
        xaxis_title="Cost per Generation ($)",
        yaxis_title="Pass Rate (%)",
        yaxis=dict(range=[0, 105]),
    )
    return apply_plotly_theme(fig)


def build_incorrect_pitches_by_model(df):
    """Build bar chart of most common incorrect pitch classes per model.

    Shows note names and counts for pitch classes that appeared incorrectly.

    Args:
        df (pd.DataFrame): Filtered results DataFrame.

    Returns:
        go.Figure: Grouped bar chart figure.
    """
    if df.empty:
        return apply_plotly_theme(go.Figure().update_layout(title="No data"))

    # Collect incorrect pitches per model
    model_pitch_counts = defaultdict(lambda: defaultdict(int))
    for _, row in df.iterrows():
        if not row["scale_ran"] or not row["scale_pitches_incorrect"]:
            continue
        model = row["model"]
        for pc in row["scale_pitches_incorrect"]:
            note = pitch_class_to_note(pc)
            model_pitch_counts[model][note] += 1

    if not model_pitch_counts:
        fig = go.Figure()
        fig.add_annotation(
            text="No incorrect pitches found",
            showarrow=False,
            font=dict(size=16, color=PLOTLY_TEXT),
        )
        fig.update_layout(title="Incorrect Pitches by Model")
        return apply_plotly_theme(fig)

    # Build grouped bar chart
    all_notes = sorted(
        set(n for counts in model_pitch_counts.values() for n in counts),
        key=lambda n: NOTE_NAMES.index(n) if n in NOTE_NAMES else 99,
    )
    fig = go.Figure()
    for model in sorted(model_pitch_counts.keys()):
        counts = model_pitch_counts[model]
        fig.add_trace(
            go.Bar(
                name=model,
                x=all_notes,
                y=[counts.get(n, 0) for n in all_notes],
                text=[counts.get(n, 0) for n in all_notes],
                textposition="auto",
            )
        )

    fig.update_layout(
        barmode="group",
        title="Most Common Incorrect Pitch Classes by Model",
        xaxis_title="Note Name",
        yaxis_title="Occurrence Count",
    )
    return apply_plotly_theme(fig)


def build_incorrect_intervals_by_model(df):
    """Build bar chart of incorrect intervals relative to prompted root, per model.

    For example, if the test was in C major and the model played Bb, that's a "m7"
    interval. This helps identify systematic errors (e.g. always confusing major/minor 3rds).

    Args:
        df (pd.DataFrame): Filtered results DataFrame.

    Returns:
        go.Figure: Grouped bar chart figure.
    """
    if df.empty:
        return apply_plotly_theme(go.Figure().update_layout(title="No data"))

    model_interval_counts = defaultdict(lambda: defaultdict(int))
    for _, row in df.iterrows():
        if not row["scale_ran"] or not row["scale_pitches_incorrect"]:
            continue
        model = row["model"]
        root_pc = note_name_to_pitch_class(row["root"])
        for pc in row["scale_pitches_incorrect"]:
            interval = pitch_class_to_interval(pc, root_pc)
            model_interval_counts[model][interval] += 1

    if not model_interval_counts:
        fig = go.Figure()
        fig.add_annotation(
            text="No incorrect intervals found",
            showarrow=False,
            font=dict(size=16, color=PLOTLY_TEXT),
        )
        fig.update_layout(title="Incorrect Intervals by Model")
        return apply_plotly_theme(fig)

    # Maintain interval order
    all_intervals = [
        iv
        for iv in INTERVAL_NAMES
        if any(iv in counts for counts in model_interval_counts.values())
    ]

    fig = go.Figure()
    for model in sorted(model_interval_counts.keys()):
        counts = model_interval_counts[model]
        fig.add_trace(
            go.Bar(
                name=model,
                x=all_intervals,
                y=[counts.get(iv, 0) for iv in all_intervals],
                text=[counts.get(iv, 0) for iv in all_intervals],
                textposition="auto",
            )
        )

    fig.update_layout(
        barmode="group",
        title="Incorrect Intervals Relative to Root (per Model)",
        xaxis_title="Interval",
        yaxis_title="Occurrence Count",
    )
    return apply_plotly_theme(fig)


def build_duration_errors_by_model(df):
    """Build bar chart of incorrect duration types per model.

    Shows what the model generated instead of the requested duration.

    Args:
        df (pd.DataFrame): Filtered results DataFrame.

    Returns:
        go.Figure: Grouped bar chart figure.
    """
    if df.empty:
        return apply_plotly_theme(go.Figure().update_layout(title="No data"))

    model_dur_counts = defaultdict(lambda: defaultdict(int))
    for _, row in df.iterrows():
        if not row["duration_ran"] or not row["duration_lengths"]:
            continue
        model = row["model"]
        requested = row["duration_param"]
        for ratio_str, count in row["duration_lengths"].items():
            name = beats_to_duration_name(float(ratio_str))
            label = f"{name} (wanted {requested})"
            model_dur_counts[model][label] += count

    if not model_dur_counts:
        fig = go.Figure()
        fig.add_annotation(
            text="No duration errors found",
            showarrow=False,
            font=dict(size=16, color=PLOTLY_TEXT),
        )
        fig.update_layout(title="Duration Errors by Model")
        return apply_plotly_theme(fig)

    all_labels = sorted({label for counts in model_dur_counts.values() for label in counts})

    fig = go.Figure()
    for model in sorted(model_dur_counts.keys()):
        counts = model_dur_counts[model]
        fig.add_trace(
            go.Bar(
                name=model,
                x=all_labels,
                y=[counts.get(label, 0) for label in all_labels],
                text=[counts.get(label, 0) for label in all_labels],
                textposition="auto",
            )
        )

    fig.update_layout(
        barmode="group",
        title="Incorrect Durations by Model (Actual vs Requested)",
        xaxis_title="Incorrect Duration (Requested)",
        yaxis_title="Note Count",
    )
    return apply_plotly_theme(fig)


def build_effort_impact_delta(df):
    """Build bar chart showing pass rate delta between lowest and highest effort per base model.

    Only includes models that have multiple effort levels in the data.
    Positive delta = higher effort helped.

    Args:
        df (pd.DataFrame): Filtered results DataFrame (must contain base_model column).

    Returns:
        go.Figure: Delta bar chart figure.
    """
    if df.empty or "base_model" not in df.columns:
        return apply_plotly_theme(go.Figure().update_layout(title="No data"))

    # Only rows that have an effort value
    effort_df = df[df["effort"].notna()].copy()
    if effort_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No effort-level data in this run",
            showarrow=False,
            font=dict(size=16, color=PLOTLY_TEXT),
        )
        fig.update_layout(title="Effort Level Impact")
        return apply_plotly_theme(fig)

    # Define a canonical effort ordering so we can identify "lowest" and "highest"
    EFFORT_ORDER = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]

    def effort_rank(e):
        e_lower = str(e).lower()
        return EFFORT_ORDER.index(e_lower) if e_lower in EFFORT_ORDER else len(EFFORT_ORDER)

    # Per (base_model, effort) pass rate
    stats = (
        effort_df.groupby(["base_model", "effort"])
        .agg(
            pass_rate=("overall_pass", "mean"),
            count=("overall_pass", "count"),
        )
        .reset_index()
    )

    models_with_levels = []
    for base in stats["base_model"].unique():
        model_stats = stats[stats["base_model"] == base].copy()
        if len(model_stats) < 2:
            continue
        model_stats["rank"] = model_stats["effort"].apply(effort_rank)
        model_stats = model_stats.sort_values("rank")
        lowest = model_stats.iloc[0]
        highest = model_stats.iloc[-1]
        delta = (highest["pass_rate"] - lowest["pass_rate"]) * 100
        models_with_levels.append(
            {
                "base_model": base,
                "lowest_effort": lowest["effort"],
                "highest_effort": highest["effort"],
                "lowest_rate": round(lowest["pass_rate"] * 100, 1),
                "highest_rate": round(highest["pass_rate"] * 100, 1),
                "delta": round(delta, 1),
            }
        )

    if not models_with_levels:
        fig = go.Figure()
        fig.add_annotation(
            text="No models with multiple effort levels",
            showarrow=False,
            font=dict(size=16, color=PLOTLY_TEXT),
        )
        fig.update_layout(title="Effort Level Impact")
        return apply_plotly_theme(fig)

    result = pd.DataFrame(models_with_levels).sort_values("delta", ascending=True)
    colors = ["#2ecc71" if d >= 0 else "#e74c3c" for d in result["delta"]]

    fig = go.Figure(
        go.Bar(
            x=result["delta"],
            y=result["base_model"],
            orientation="h",
            text=[
                f"{d:+.1f}pp ({lo}% @ {le} -> {hi}% @ {he})"
                for d, lo, le, hi, he in zip(
                    result["delta"],
                    result["lowest_rate"],
                    result["lowest_effort"],
                    result["highest_rate"],
                    result["highest_effort"],
                )
            ],
            textposition="auto",
            marker_color=colors,
        )
    )
    fig.update_layout(
        title="Effort Level Impact (Highest - Lowest Effort)",
        xaxis_title="Pass Rate Delta (pp)",
        yaxis_title="",
    )
    fig.add_vline(x=0, line_dash="dash", line_color=PLOTLY_TEXT, opacity=0.5)
    return apply_plotly_theme(fig)


def build_reasoning_toggle_comparison(df):
    """Build grouped bar chart comparing standard vs reasoning-enabled for legacy toggle models.

    Legacy models are those with use_thinking toggle but no effort levels
    (e.g. claude-opus-4-5, gemini-2.5-pro).

    Args:
        df (pd.DataFrame): Filtered results DataFrame (must contain base_model column).

    Returns:
        go.Figure: Grouped bar chart figure.
    """
    if df.empty or "base_model" not in df.columns:
        return apply_plotly_theme(go.Figure().update_layout(title="No data"))

    # Find legacy toggle models: base_models that have both use_thinking=True and
    # use_thinking=False rows, but NO effort values
    toggle_models = []
    for base in df["base_model"].unique():
        mask = df["base_model"] == base
        subset = df.loc[mask]
        no_effort = subset["effort"].isna().all()
        has_thinking = subset["use_thinking"].any()
        has_standard = not subset["use_thinking"].all()
        if no_effort and has_thinking and has_standard:
            toggle_models.append(base)

    if not toggle_models:
        fig = go.Figure()
        fig.add_annotation(
            text="No toggle-based reasoning models in this run",
            showarrow=False,
            font=dict(size=16, color=PLOTLY_TEXT),
        )
        fig.update_layout(title="Standard vs Reasoning Toggle")
        return apply_plotly_theme(fig)

    toggle_models = sorted(toggle_models)
    std_rates = []
    reas_rates = []
    std_latencies = []
    reas_latencies = []
    std_costs = []
    reas_costs = []

    for base in toggle_models:
        mask = df["base_model"] == base
        std = df.loc[mask & ~df["use_thinking"]]
        reas = df.loc[mask & df["use_thinking"]]
        std_rates.append(round(std["overall_pass"].mean() * 100, 1) if len(std) > 0 else 0)
        reas_rates.append(round(reas["overall_pass"].mean() * 100, 1) if len(reas) > 0 else 0)
        std_latencies.append(round(std["api_latency"].mean(), 1) if len(std) > 0 else 0)
        reas_latencies.append(round(reas["api_latency"].mean(), 1) if len(reas) > 0 else 0)
        std_costs.append(round(std["cost"].mean(), 5) if len(std) > 0 else 0)
        reas_costs.append(round(reas["cost"].mean(), 5) if len(reas) > 0 else 0)

    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["Pass Rate (%)", "Avg Latency (s)", "Avg Cost ($)"],
        shared_yaxes=True,
    )

    # Pass Rate
    fig.add_trace(
        go.Bar(
            name="Standard",
            x=std_rates,
            y=toggle_models,
            orientation="h",
            text=[f"{v}%" for v in std_rates],
            textposition="auto",
            marker_color="#5dade2",
            showlegend=True,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            name="Reasoning",
            x=reas_rates,
            y=toggle_models,
            orientation="h",
            text=[f"{v}%" for v in reas_rates],
            textposition="auto",
            marker_color="#f39c12",
            showlegend=True,
        ),
        row=1,
        col=1,
    )
    # Latency
    fig.add_trace(
        go.Bar(
            x=std_latencies,
            y=toggle_models,
            orientation="h",
            text=[f"{v}s" for v in std_latencies],
            textposition="auto",
            marker_color="#5dade2",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=reas_latencies,
            y=toggle_models,
            orientation="h",
            text=[f"{v}s" for v in reas_latencies],
            textposition="auto",
            marker_color="#f39c12",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    # Cost
    fig.add_trace(
        go.Bar(
            x=std_costs,
            y=toggle_models,
            orientation="h",
            text=[f"${v:.5f}" for v in std_costs],
            textposition="auto",
            marker_color="#5dade2",
            showlegend=False,
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Bar(
            x=reas_costs,
            y=toggle_models,
            orientation="h",
            text=[f"${v:.5f}" for v in reas_costs],
            textposition="auto",
            marker_color="#f39c12",
            showlegend=False,
        ),
        row=1,
        col=3,
    )

    fig.update_layout(
        title="Standard vs Reasoning Toggle (Legacy Models)",
        barmode="group",
        height=max(300, len(toggle_models) * 80 + 150),
    )
    return apply_plotly_theme(fig)


def build_reasoning_cost_effectiveness(df):
    """Build scatter plot of cost per generation vs pass rate with effort trajectory lines.

    Each model+effort instance is a point. Points belonging to the same base model
    are connected with a line to show the cost/quality tradeoff as effort increases.

    Args:
        df (pd.DataFrame): Filtered results DataFrame (must contain base_model column).

    Returns:
        go.Figure: Scatter plot figure.
    """
    if df.empty or "base_model" not in df.columns:
        return apply_plotly_theme(go.Figure().update_layout(title="No data"))

    stats = (
        df.groupby(["model", "base_model"])
        .agg(
            total_cost=("cost", "sum"),
            tested=("cost", "count"),
            pass_rate=("overall_pass", "mean"),
            avg_latency=("api_latency", "mean"),
        )
        .reset_index()
    )

    if stats["total_cost"].sum() == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="All costs are $0 (local models)",
            showarrow=False,
            font=dict(size=16, color=PLOTLY_TEXT),
        )
        fig.update_layout(title="Reasoning Cost-Effectiveness")
        return apply_plotly_theme(fig)

    stats["cost_per_gen"] = stats["total_cost"] / stats["tested"]
    stats["pass_rate_pct"] = (stats["pass_rate"] * 100).round(1)

    EFFORT_ORDER = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]

    # Assign colors per base_model
    base_models = sorted(stats["base_model"].unique())
    color_map = {bm: MODEL_COLORS[i % len(MODEL_COLORS)] for i, bm in enumerate(base_models)}

    fig = go.Figure()

    # Draw trajectory lines connecting effort levels of the same base model
    for base in base_models:
        bm_stats = stats[stats["base_model"] == base].copy()
        if len(bm_stats) > 1:
            # Extract effort from the instance name and sort by effort order
            def _effort_from_name(name):
                if "(" in name and ")" in name:
                    e = name.split("(")[-1].rstrip(")")
                    return EFFORT_ORDER.index(e) if e in EFFORT_ORDER else len(EFFORT_ORDER)
                return -1

            bm_stats["_rank"] = bm_stats["model"].apply(_effort_from_name)
            bm_stats = bm_stats.sort_values("_rank")
            fig.add_trace(
                go.Scatter(
                    x=bm_stats["cost_per_gen"],
                    y=bm_stats["pass_rate_pct"],
                    mode="lines",
                    line=dict(color=color_map[base], width=1.5, dash="dot"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    # Compute text positions across ALL points before splitting into per-model traces
    all_text_positions = compute_text_positions(stats["cost_per_gen"], stats["pass_rate_pct"])
    stats["_text_pos"] = all_text_positions

    # Draw scatter points
    for base in base_models:
        bm_stats = stats[stats["base_model"] == base]
        fig.add_trace(
            go.Scatter(
                x=bm_stats["cost_per_gen"],
                y=bm_stats["pass_rate_pct"],
                mode="markers+text",
                name=base,
                text=bm_stats["model"],
                textposition=list(bm_stats["_text_pos"]),
                marker=dict(size=12, color=color_map[base]),
            )
        )

    fig.update_layout(
        title="Reasoning Cost-Effectiveness (Cost per Generation vs Pass Rate)",
        xaxis_title="Cost per Generation ($)",
        yaxis_title="Pass Rate (%)",
        yaxis=dict(range=[0, 105]),
    )
    return apply_plotly_theme(fig)


def build_failure_rate_by_model(df):
    """Build bar chart of generation failure rate (errors) per model.

    Args:
        df (pd.DataFrame): Filtered results DataFrame.

    Returns:
        go.Figure: Bar chart figure.
    """
    if df.empty:
        return apply_plotly_theme(go.Figure().update_layout(title="No data"))

    stats = (
        df.groupby("model")
        .agg(
            total=("has_error", "count"),
            errors=("has_error", "sum"),
        )
        .reset_index()
    )
    stats["error_rate"] = (stats["errors"] / stats["total"] * 100).round(1)
    stats = stats.sort_values("error_rate", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=stats["error_rate"],
            y=stats["model"],
            orientation="h",
            text=[
                f"{r}% ({e}/{t})"
                for r, e, t in zip(
                    stats["error_rate"],
                    stats["errors"].astype(int),
                    stats["total"].astype(int),
                )
            ],
            textposition="auto",
            marker_color="#e74c3c",
        )
    )
    fig.update_layout(
        title="Generation Failure Rate by Model",
        xaxis_title="Failure Rate (%)",
        yaxis_title="",
        xaxis=dict(range=[0, max(stats["error_rate"].max() * 1.2, 10)]),
    )
    return apply_plotly_theme(fig)


def make_metric_card(title, value, subtitle="", color="#5dade2"):
    """Create a Bootstrap-styled metric card.

    Args:
        title (str): Card title.
        value (str): Main value to display.
        subtitle (str): Optional subtitle text.
        color (str): Accent color for the value.

    Returns:
        dbc.Card: Dash Bootstrap card component.
    """
    return dbc.Card(
        dbc.CardBody(
            [
                html.P(
                    title,
                    className="card-title mb-1",
                    style={
                        "color": "#999",
                        "fontSize": "0.85rem",
                        "textTransform": "uppercase",
                    },
                ),
                html.H3(
                    value,
                    className="mb-0",
                    style={"color": color, "fontWeight": "bold"},
                ),
                html.Small(subtitle, style={"color": "#666"}) if subtitle else None,
            ]
        ),
        style={
            "backgroundColor": PLOTLY_CARD_BG,
            "border": f"1px solid {PLOTLY_ACCENT}",
            "borderRadius": "8px",
        },
        className="h-100",
    )


def build_filter_bar(df):
    """Build the global filter controls row.

    Args:
        df (pd.DataFrame): Full results DataFrame.

    Returns:
        dbc.Row: Row of filter dropdowns.
    """
    models = sorted(df["model"].unique()) if not df.empty else []
    roots = sorted(df["root"].unique()) if not df.empty else []
    scales = sorted(df["scale"].unique()) if not df.empty else []
    variations = sorted(df["variation"].unique()) if not df.empty else []

    return dbc.Row(
        [
            dbc.Col(
                [
                    html.Label("Models", style={"color": PLOTLY_TEXT, "fontSize": "0.8rem"}),
                    dcc.Dropdown(
                        id="filter-models",
                        options=[{"label": m, "value": m} for m in models],
                        value=models,
                        multi=True,
                        style={"backgroundColor": PLOTLY_CARD_BG, "color": "#000"},
                    ),
                ],
                md=4,
            ),
            dbc.Col(
                [
                    html.Label("Root Notes", style={"color": PLOTLY_TEXT, "fontSize": "0.8rem"}),
                    dcc.Dropdown(
                        id="filter-roots",
                        options=[{"label": r, "value": r} for r in roots],
                        value=roots,
                        multi=True,
                        style={"backgroundColor": PLOTLY_CARD_BG, "color": "#000"},
                    ),
                ],
                md=2,
            ),
            dbc.Col(
                [
                    html.Label("Scale Type", style={"color": PLOTLY_TEXT, "fontSize": "0.8rem"}),
                    dcc.Dropdown(
                        id="filter-scales",
                        options=[{"label": s.title(), "value": s} for s in scales],
                        value=scales,
                        multi=True,
                        style={"backgroundColor": PLOTLY_CARD_BG, "color": "#000"},
                    ),
                ],
                md=2,
            ),
            dbc.Col(
                [
                    html.Label("Variation", style={"color": PLOTLY_TEXT, "fontSize": "0.8rem"}),
                    dcc.Dropdown(
                        id="filter-variations",
                        options=[
                            {"label": v.replace("_", " ").title(), "value": v} for v in variations
                        ],
                        value=variations,
                        multi=True,
                        style={"backgroundColor": PLOTLY_CARD_BG, "color": "#000"},
                    ),
                ],
                md=3,
            ),
            dbc.Col(
                [
                    html.Label("\u00a0", style={"fontSize": "0.8rem"}),
                    html.Div(
                        dbc.Button(
                            "Export Dashboard",
                            id="export-btn",
                            color="secondary",
                            size="sm",
                            className="w-100",
                        ),
                    ),
                ],
                md=1,
            ),
        ],
        className="mb-3 g-2",
    )


def create_app(run_path):
    """Create and configure the Dash application.

    The Reasoning tab is only included when the run's config enables it.

    Args:
        run_path (str): Path to the evaluation run directory.

    Returns:
        dash.Dash: Configured Dash application.
    """
    df, config, summary = load_run(run_path)

    if df.empty:
        print(f"No results found in {run_path}")
        sys.exit(1)

    run_name = config.get("run_name", Path(run_path).name)
    timestamp = config.get("timestamp", "")
    totals = summary.get("totals", {})
    has_reasoning = config.get("test_reasoning", False)

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        title=f"LoopGPT Eval: {run_name}",
    )

    # Build tab list conditionally based on run feature flags
    tab_list = [
        # ── Tab 1: Overview ──
        dbc.Tab(
            label="Overview",
            tab_id="tab-overview",
            children=[
                html.Div(id="tab-overview-content", className="mt-3"),
            ],
        ),
        # ── Tab 2: Model Performance ──
        dbc.Tab(
            label="Model Performance",
            tab_id="tab-model",
            children=[
                html.Div(id="tab-model-content", className="mt-3"),
            ],
        ),
        # ── Tab 3: Root & Scale ──
        dbc.Tab(
            label="Root & Scale",
            tab_id="tab-root-scale",
            children=[
                html.Div(id="tab-root-scale-content", className="mt-3"),
            ],
        ),
    ]

    tab_list.extend(
        [
            # ── Latency ──
            dbc.Tab(
                label="Latency",
                tab_id="tab-latency",
                children=[
                    html.Div(id="tab-latency-content", className="mt-3"),
                ],
            ),
            # ── Cost ──
            dbc.Tab(
                label="Cost",
                tab_id="tab-cost",
                children=[
                    html.Div(id="tab-cost-content", className="mt-3"),
                ],
            ),
        ]
    )

    if has_reasoning:
        tab_list.append(
            dbc.Tab(
                label="Reasoning",
                tab_id="tab-reasoning",
                children=[
                    html.Div(id="tab-reasoning-content", className="mt-3"),
                ],
            ),
        )

    tab_list.append(
        # ── Error Patterns (always last) ──
        dbc.Tab(
            label="Error Patterns",
            tab_id="tab-errors",
            children=[
                html.Div(id="tab-errors-content", className="mt-3"),
            ],
        ),
    )

    app.layout = dbc.Container(
        [
            # Hidden store for the full data
            dcc.Store(id="run-data", data=df.to_json(date_format="iso", orient="split")),
            # Header
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H2(
                                f"LoopGPT Evaluation: {run_name}",
                                style={"color": PLOTLY_TEXT, "fontWeight": "bold"},
                            ),
                            html.P(
                                f"Run: {timestamp}  |  {len(df)} total generations  |  "
                                f"{len(df['model'].unique())} models  |  "
                                f"{len(df['root'].unique())} roots  |  "
                                f"{len(df['original_prompt'].unique())} prompts",
                                style={"color": "#999"},
                            ),
                        ]
                    ),
                ],
                className="mb-3 mt-3",
            ),
            # Global filters
            build_filter_bar(df),
            # Export status
            html.Div(id="export-status", className="mb-2"),
            # Tabs (conditionally built)
            dbc.Tabs(tab_list, id="tabs", active_tab="tab-overview", className="mb-3"),
        ],
        fluid=True,
        style={"backgroundColor": PLOTLY_BG, "minHeight": "100vh", "padding": "20px"},
    )

    # ── Callbacks ──

    @app.callback(
        Output("tab-overview-content", "children"),
        [
            Input("filter-models", "value"),
            Input("filter-roots", "value"),
            Input("filter-scales", "value"),
            Input("filter-variations", "value"),
        ],
    )
    def update_overview(models, roots, scales, variations):
        filtered = apply_filters(df, models, roots, scales, variations)
        if filtered.empty:
            return html.P("No data matches current filters.", style={"color": PLOTLY_TEXT})

        total = len(filtered)
        passed = int(filtered["overall_pass"].sum())
        failed_gen = int(filtered["has_error"].sum())
        pass_rate = round(passed / total * 100, 1) if total > 0 else 0
        total_cost = filtered["cost"].sum()
        avg_latency = filtered["api_latency"].mean()

        # Best / worst model
        model_rates = filtered.groupby("model")["overall_pass"].mean().sort_values(ascending=False)
        best_model = (
            f"{model_rates.index[0]} ({model_rates.iloc[0] * 100:.1f}%)"
            if len(model_rates) > 0
            else "N/A"
        )
        worst_model = (
            f"{model_rates.index[-1]} ({model_rates.iloc[-1] * 100:.1f}%)"
            if len(model_rates) > 1
            else "N/A"
        )

        return html.Div(
            [
                # Metric cards row
                dbc.Row(
                    [
                        dbc.Col(make_metric_card("Total Generations", str(total)), md=2),
                        dbc.Col(
                            make_metric_card(
                                "Pass Rate",
                                f"{pass_rate}%",
                                f"{passed} passed / {total - passed - failed_gen} failed / {failed_gen} errors",
                                color="#2ecc71" if pass_rate >= 50 else "#e74c3c",
                            ),
                            md=2,
                        ),
                        dbc.Col(
                            make_metric_card("Best Model", best_model, color="#2ecc71"),
                            md=2,
                        ),
                        dbc.Col(
                            make_metric_card("Worst Model", worst_model, color="#e74c3c"),
                            md=2,
                        ),
                        dbc.Col(make_metric_card("Total Cost", f"${total_cost:.4f}"), md=2),
                        dbc.Col(make_metric_card("Avg Latency", f"{avg_latency:.1f}s"), md=2),
                    ],
                    className="mb-4 g-2",
                ),
                # Main chart
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(figure=build_pass_rate_by_model(filtered)), md=12),
                    ]
                ),
            ]
        )

    @app.callback(
        Output("tab-model-content", "children"),
        [
            Input("filter-models", "value"),
            Input("filter-roots", "value"),
            Input("filter-scales", "value"),
            Input("filter-variations", "value"),
        ],
    )
    def update_model(models, roots, scales, variations):
        filtered = apply_filters(df, models, roots, scales, variations)
        if filtered.empty:
            return html.P("No data matches current filters.", style={"color": PLOTLY_TEXT})

        return html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(figure=build_per_test_breakdown(filtered)), md=12),
                    ],
                    className="mb-3",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Graph(figure=build_major_vs_minor_by_model(filtered)),
                            md=6,
                        ),
                        dbc.Col(dcc.Graph(figure=build_model_scale_heatmap(filtered)), md=6),
                    ],
                    className="mb-3",
                ),
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(figure=build_model_root_heatmap(filtered)), md=12),
                    ]
                ),
            ]
        )

    @app.callback(
        Output("tab-root-scale-content", "children"),
        [
            Input("filter-models", "value"),
            Input("filter-roots", "value"),
            Input("filter-scales", "value"),
            Input("filter-variations", "value"),
        ],
    )
    def update_root_scale(models, roots, scales, variations):
        filtered = apply_filters(df, models, roots, scales, variations)
        if filtered.empty:
            return html.P("No data matches current filters.", style={"color": PLOTLY_TEXT})

        return html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(figure=build_root_pass_rate(filtered)), md=6),
                        dbc.Col(dcc.Graph(figure=build_root_scale_grouped(filtered)), md=6),
                    ],
                    className="mb-3",
                ),
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(figure=build_root_scale_heatmap(filtered)), md=12),
                    ]
                ),
            ]
        )

    @app.callback(  # noqa: E303
        Output("tab-latency-content", "children"),
        [
            Input("filter-models", "value"),
            Input("filter-roots", "value"),
            Input("filter-scales", "value"),
            Input("filter-variations", "value"),
        ],
    )
    def update_latency(models, roots, scales, variations):
        filtered = apply_filters(df, models, roots, scales, variations)
        if filtered.empty:
            return html.P("No data matches current filters.", style={"color": PLOTLY_TEXT})

        return html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(figure=build_latency_box(filtered)), md=12),
                    ],
                    className="mb-3",
                ),
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(figure=build_latency_vs_pass(filtered)), md=12),
                    ]
                ),
            ]
        )

    @app.callback(
        Output("tab-cost-content", "children"),
        [
            Input("filter-models", "value"),
            Input("filter-roots", "value"),
            Input("filter-scales", "value"),
            Input("filter-variations", "value"),
        ],
    )
    def update_cost(models, roots, scales, variations):
        filtered = apply_filters(df, models, roots, scales, variations)
        if filtered.empty:
            return html.P("No data matches current filters.", style={"color": PLOTLY_TEXT})

        return html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(figure=build_cost_by_model(filtered)), md=6),
                        dbc.Col(dcc.Graph(figure=build_cost_vs_pass(filtered)), md=6),
                    ]
                ),
            ]
        )

    # ── Reasoning callback (only registered when reasoning was tested) ──
    if has_reasoning:

        @app.callback(
            Output("tab-reasoning-content", "children"),
            [
                Input("filter-models", "value"),
                Input("filter-roots", "value"),
                Input("filter-scales", "value"),
                Input("filter-variations", "value"),
            ],
        )
        def update_reasoning(models, roots, scales, variations):
            filtered = apply_filters(df, models, roots, scales, variations)
            if filtered.empty:
                return html.P("No data matches current filters.", style={"color": PLOTLY_TEXT})

            # Summary cards for reasoning impact
            effort_rows = filtered[filtered["effort"].notna()]
            toggle_rows = filtered[
                filtered["effort"].isna()
                & filtered["base_model"].isin(
                    [
                        bm
                        for bm in filtered["base_model"].unique()
                        if not filtered.loc[filtered["base_model"] == bm, "use_thinking"].all()
                        and filtered.loc[filtered["base_model"] == bm, "use_thinking"].any()
                        and filtered.loc[filtered["base_model"] == bm, "effort"].isna().all()
                    ]
                )
            ]
            n_effort_models = effort_rows["base_model"].nunique() if not effort_rows.empty else 0
            n_toggle_models = toggle_rows["base_model"].nunique() if not toggle_rows.empty else 0

            cards = dbc.Row(
                [
                    dbc.Col(
                        make_metric_card(
                            "Effort-Based Models",
                            str(n_effort_models),
                            "Models with effort level options",
                        ),
                        md=3,
                    ),
                    dbc.Col(
                        make_metric_card(
                            "Toggle-Based Models",
                            str(n_toggle_models),
                            "Legacy models with on/off reasoning",
                        ),
                        md=3,
                    ),
                    dbc.Col(
                        make_metric_card(
                            "Total Instances",
                            str(filtered["model"].nunique()),
                            "Unique model + effort combinations",
                        ),
                        md=3,
                    ),
                    dbc.Col(
                        make_metric_card(
                            "Avg Pass Rate",
                            f"{filtered['overall_pass'].mean() * 100:.1f}%",
                        ),
                        md=3,
                    ),
                ],
                className="mb-4 g-2",
            )

            rows = [cards]

            # Effort impact delta chart
            rows.append(
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(figure=build_effort_impact_delta(filtered)), md=12),
                    ],
                    className="mb-3",
                )
            )

            # Toggle comparison chart
            rows.append(
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Graph(figure=build_reasoning_toggle_comparison(filtered)),
                            md=12,
                        ),
                    ],
                    className="mb-3",
                )
            )

            # Cost-effectiveness scatter
            rows.append(
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Graph(figure=build_reasoning_cost_effectiveness(filtered)),
                            md=12,
                        ),
                    ]
                )
            )

            return html.Div(rows)

    @app.callback(
        Output("tab-errors-content", "children"),
        [
            Input("filter-models", "value"),
            Input("filter-roots", "value"),
            Input("filter-scales", "value"),
            Input("filter-variations", "value"),
        ],
    )
    def update_errors(models, roots, scales, variations):
        filtered = apply_filters(df, models, roots, scales, variations)
        if filtered.empty:
            return html.P("No data matches current filters.", style={"color": PLOTLY_TEXT})

        return html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Graph(figure=build_failure_rate_by_model(filtered)),
                            md=12,
                        ),
                    ],
                    className="mb-3",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Graph(figure=build_incorrect_pitches_by_model(filtered)),
                            md=6,
                        ),
                        dbc.Col(
                            dcc.Graph(figure=build_incorrect_intervals_by_model(filtered)),
                            md=6,
                        ),
                    ],
                    className="mb-3",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Graph(figure=build_duration_errors_by_model(filtered)),
                            md=12,
                        ),
                    ]
                ),
            ]
        )

    @app.callback(
        Output("export-status", "children"),
        Input("export-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def export_dashboard(n_clicks):
        if not n_clicks:
            return ""

        export_dir = Path(run_path) / "analysis"
        export_dir.mkdir(exist_ok=True)

        # Build all figures with the full (unfiltered) dataset
        figures = {
            "pass_rate_by_model": build_pass_rate_by_model(df),
            "per_test_breakdown": build_per_test_breakdown(df),
            "model_root_heatmap": build_model_root_heatmap(df),
            "model_scale_heatmap": build_model_scale_heatmap(df),
            "major_vs_minor": build_major_vs_minor_by_model(df),
            "root_pass_rate": build_root_pass_rate(df),
            "root_scale_grouped": build_root_scale_grouped(df),
            "root_scale_heatmap": build_root_scale_heatmap(df),
            "latency_box": build_latency_box(df),
            "latency_vs_pass": build_latency_vs_pass(df),
            "cost_by_model": build_cost_by_model(df),
            "cost_vs_pass": build_cost_vs_pass(df),
            "failure_rate": build_failure_rate_by_model(df),
            "incorrect_pitches": build_incorrect_pitches_by_model(df),
            "incorrect_intervals": build_incorrect_intervals_by_model(df),
            "duration_errors": build_duration_errors_by_model(df),
        }

        if has_reasoning:
            figures["effort_impact_delta"] = build_effort_impact_delta(df)
            figures["reasoning_toggle"] = build_reasoning_toggle_comparison(df)
            figures["reasoning_cost_effectiveness"] = build_reasoning_cost_effectiveness(df)

        # Save individual charts
        for name, fig in figures.items():
            fig.write_html(str(export_dir / f"{name}.html"), include_plotlyjs="cdn")

        # Build combined dashboard HTML
        combined_html = _build_combined_html(figures, run_name, timestamp, totals, df)
        combined_path = export_dir / "dashboard.html"
        with open(combined_path, "w", encoding="utf-8") as f:
            f.write(combined_html)

        return dbc.Alert(
            f"Dashboard exported to {export_dir}/  ({len(figures)} charts + dashboard.html)",
            color="success",
            dismissable=True,
            duration=5000,
        )

    return app


def _build_combined_html(figures, run_name, timestamp, totals, df):
    """Build a single self-contained HTML page with all charts.

    Args:
        figures (dict): Dictionary of name -> go.Figure.
        run_name (str): Run name for the header.
        timestamp (str): Run timestamp.
        totals (dict): Summary totals dict.
        df (pd.DataFrame): Full results DataFrame.

    Returns:
        str: Complete HTML string.
    """
    total = len(df)
    passed = int(df["overall_pass"].sum())
    pass_rate = round(passed / total * 100, 1) if total > 0 else 0

    chart_divs = []
    for name, fig in figures.items():
        title = name.replace("_", " ").title()
        div_html = fig.to_html(include_plotlyjs=False, full_html=False, div_id=f"chart-{name}")
        chart_divs.append(f'<div class="chart-section"><h3>{title}</h3>{div_html}</div>')

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>LoopGPT Eval Dashboard: {run_name}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ background-color: {PLOTLY_BG}; color: {PLOTLY_TEXT}; font-family: 'Segoe UI', sans-serif; margin: 0; padding: 20px; }}
        h1 {{ color: #5dade2; }}
        h3 {{ color: #999; border-bottom: 1px solid {PLOTLY_ACCENT}; padding-bottom: 8px; }}
        .stats {{ display: flex; gap: 20px; margin-bottom: 30px; flex-wrap: wrap; }}
        .stat-card {{ background: {PLOTLY_CARD_BG}; border: 1px solid {PLOTLY_ACCENT}; border-radius: 8px; padding: 15px 20px; min-width: 150px; }}
        .stat-card .label {{ color: #999; font-size: 0.8rem; text-transform: uppercase; }}
        .stat-card .value {{ font-size: 1.5rem; font-weight: bold; color: #5dade2; }}
        .chart-section {{ margin-bottom: 40px; }}
    </style>
</head>
<body>
    <h1>LoopGPT Evaluation Dashboard: {run_name}</h1>
    <p style="color: #666">Run: {timestamp} | {total} generations | {len(df["model"].unique())} models</p>
    <div class="stats">
        <div class="stat-card"><div class="label">Total</div><div class="value">{total}</div></div>
        <div class="stat-card"><div class="label">Pass Rate</div><div class="value" style="color: {"#2ecc71" if pass_rate >= 50 else "#e74c3c"}">{pass_rate}%</div></div>
        <div class="stat-card"><div class="label">Passed</div><div class="value">{passed}</div></div>
        <div class="stat-card"><div class="label">Total Cost</div><div class="value">${df["cost"].sum():.4f}</div></div>
    </div>
    {"".join(chart_divs)}
</body>
</html>"""


def main():
    """Entry point. Parses CLI args or prompts for run selection, then launches the dashboard."""
    if len(sys.argv) > 1:
        run_path = sys.argv[1]
        # Support both full path and just run name
        if not os.path.isdir(run_path):
            # Try the evaluator's default output directory.
            candidate = DEFAULT_EVALUATIONS_DIR / run_path
            if os.path.isdir(candidate):
                run_path = candidate
            else:
                print(f"Run directory not found: {run_path}")
                sys.exit(1)
    else:
        run_path = select_run_interactive()

    run_path = str(run_path)
    print(f"\nLoading evaluation run: {run_path}")

    app = create_app(run_path)

    print("\nStarting dashboard at http://127.0.0.1:8050/")
    print("Press Ctrl+C to stop.\n")
    app.run(debug=False, port=8050)


if __name__ == "__main__":
    main()
