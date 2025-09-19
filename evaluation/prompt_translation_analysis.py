from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

def safe_div(n, d):
    return (n / d) * 100.0 if d else 0.0

def annotate_bars(ax, rects, fmt="{:.1f}%"):
    for r in rects:
        height = r.get_height()
        ax.annotate(fmt.format(height),
                    xy=(r.get_x() + r.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

def annotate_bars_counts(ax, rects):
    for r in rects:
        height = r.get_height()
        ax.annotate(f"{int(height)}",
                    xy=(r.get_x() + r.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

def main():
    base = Path(__file__).resolve().parents[1]
    log_path = base / "evaluation_log.json"
    results_dir = base / "evaluation" / "Results"
    os.makedirs(results_dir, exist_ok=True)

    with open(log_path, "r", encoding="utf-8") as f:
        eval_log = json.load(f)

    # Aggregate by translation status (False = standard, True = translated)
    group = {
        False: {"total": 0, "in_key": 0, "note_duration": 0, "all_correct": 0, "latency_sum": 0.0, "cost_sum": 0.0},
        True:  {"total": 0, "in_key": 0, "note_duration": 0, "all_correct": 0, "latency_sum": 0.0, "cost_sum": 0.0},
    }

    # Aggregate by model and translation status
    model_stats = defaultdict(lambda: {
        False: {"total": 0, "in_key": 0, "note_duration": 0, "all_correct": 0},
        True:  {"total": 0, "in_key": 0, "note_duration": 0, "all_correct": 0},
    })

    for t in eval_log:
        tr = bool(t.get("translated", False))
        group[tr]["total"] += 1
        group[tr]["in_key"] += 1 if t.get("in_key_pass") else 0
        group[tr]["note_duration"] += 1 if t.get("note_length_pass") else 0
        group[tr]["all_correct"] += 1 if t.get("output_pass") else 0
        group[tr]["latency_sum"] += float(t.get("api_latency", 0.0) or 0.0)
        group[tr]["cost_sum"] += float(t.get("cost", 0.0) or 0.0)

        m = t.get("model", "unknown")
        model_stats[m][tr]["total"] += 1
        model_stats[m][tr]["in_key"] += 1 if t.get("in_key_pass") else 0
        model_stats[m][tr]["note_duration"] += 1 if t.get("note_length_pass") else 0
        model_stats[m][tr]["all_correct"] += 1 if t.get("output_pass") else 0

    # 1) Overall accuracy: translated vs standard
    labels = ["In Key", "Note Duration", "All Correct"]
    std_vals = [
        safe_div(group[False]["in_key"], group[False]["total"]),
        safe_div(group[False]["note_duration"], group[False]["total"]),
        safe_div(group[False]["all_correct"], group[False]["total"]),
    ]
    tr_vals = [
        safe_div(group[True]["in_key"], group[True]["total"]),
        safe_div(group[True]["note_duration"], group[True]["total"]),
        safe_div(group[True]["all_correct"], group[True]["total"]),
    ]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 5))
    r1 = plt.bar(x - width/2, std_vals, width, label="Standard (Untranslated)")
    r2 = plt.bar(x + width/2, tr_vals, width, label="Translated")
    plt.ylabel("Accuracy (%)")
    plt.title("Overall Accuracy: Standard vs Translated")
    plt.xticks(x, labels)
    plt.ylim(0, 105)
    annotate_bars(plt.gca(), r1)
    annotate_bars(plt.gca(), r2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "overall_accuracy_translated_vs_standard.png")
    plt.show()

    # 2) Per-model: All Correct by translation status
    models = sorted(model_stats.keys())
    std_model_vals = []
    tr_model_vals = []
    model_counts_std = []
    model_counts_tr = []

    for m in models:
        std_total = model_stats[m][False]["total"]
        tr_total = model_stats[m][True]["total"]
        std_model_vals.append(safe_div(model_stats[m][False]["all_correct"], std_total))
        tr_model_vals.append(safe_div(model_stats[m][True]["all_correct"], tr_total))
        model_counts_std.append(std_total)
        model_counts_tr.append(tr_total)

    x = np.arange(len(models))
    width = 0.38

    plt.figure(figsize=(max(10, len(models) * 0.6), 6))
    r1 = plt.bar(x - width/2, std_model_vals, width, label="Standard")
    r2 = plt.bar(x + width/2, tr_model_vals, width, label="Translated")
    plt.ylabel("All Correct (%)")
    plt.title("All Correct by Model: Standard vs Translated")
    plt.xticks(x, models, rotation=45, ha="right")
    plt.ylim(0, 105)
    annotate_bars(plt.gca(), r1)
    annotate_bars(plt.gca(), r2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "all_correct_by_model_translated_vs_standard.png")
    plt.show()

    # 3) Avg latency and cost by translation status
    avg_latency_std = (group[False]["latency_sum"] / group[False]["total"]) if group[False]["total"] else 0.0
    avg_latency_tr = (group[True]["latency_sum"] / group[True]["total"]) if group[True]["total"] else 0.0
    avg_cost_std = (group[False]["cost_sum"] / group[False]["total"]) if group[False]["total"] else 0.0
    avg_cost_tr = (group[True]["cost_sum"] / group[True]["total"]) if group[True]["total"] else 0.0

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    # Latency
    rlat = axs[0].bar(["Standard", "Translated"], [avg_latency_std, avg_latency_tr], color=["#4C78A8", "#F58518"])
    axs[0].set_title("Avg Latency by Translation Status")
    axs[0].set_ylabel("Seconds")
    annotate_bars(axs[0], rlat, fmt="{:.2f}")

    # Cost
    rcost = axs[1].bar(["Standard", "Translated"], [avg_cost_std, avg_cost_tr], color=["#4C78A8", "#F58518"])
    axs[1].set_title("Avg Cost by Translation Status")
    axs[1].set_ylabel("USD")
    annotate_bars(axs[1], rcost, fmt="${:.5f}")

    plt.tight_layout()
    fig.savefig(results_dir / "latency_cost_by_translation.png")
    plt.show()

    # Console summary
    print("=== Overall (Standard vs Translated) ===")
    print(f"Standard: N={group[False]['total']}, In Key={std_vals[0]:.1f}%, Note Dur={std_vals[1]:.1f}%, All Correct={std_vals[2]:.1f}%, "
          f"Avg Lat={avg_latency_std:.2f}s, Avg Cost=${avg_cost_std:.5f}")
    print(f"Translated: N={group[True]['total']}, In Key={tr_vals[0]:.1f}%, Note Dur={tr_vals[1]:.1f}%, All Correct={tr_vals[2]:.1f}%, "
          f"Avg Lat={avg_latency_tr:.2f}s, Avg Cost=${avg_cost_tr:.5f}")

if __name__ == "__main__":
    main()