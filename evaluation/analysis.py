import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def load_data(filepath="evaluation_log.json"):
    """Load and return evaluation data"""
    with open(filepath, "r") as f:
        return json.load(f)

def standardize_model_name(model, reasoning):
    """Standardize model names by removing dates and adding reasoning flag"""
    date = model.split("-")[-1]
    if date.isdigit() and date != "5":
        model = model[: -(len(date) + 1)]
    if reasoning == "True":
        model += " w/ reasoning"
    return model

def extract_key_info_from_path(file_path):
    """Extract key and duration info from file path"""
    # Example: "A_Major_arpeggiator_using_only_eighth_notes_False.mid"
    filename = file_path.split("\\")[-1]
    parts = filename.split("_")
    
    root = parts[0] # A, Bb, C#, etc.
    scale_type = parts[1]  # Major, Minor
    duration = parts[6]  # eighth, quarter 
    
    return root, scale_type, duration

def analyze_basic_performance(eval_log):
    """Your existing analysis - overall pass rates and scale accuracy"""
    model_pass_counts = defaultdict(int)
    model_total_counts = defaultdict(int)
    scale_accuracy = defaultdict(lambda: {"total_notes": 0, "diatonic": 0, "non_diatonic": 0})
    key_performance = defaultdict(lambda: {"Major": {"pass": 0, "total": 0}, "Minor": {"pass": 0, "total": 0}})

    for entry in eval_log:
        model = standardize_model_name(entry["model"], entry["reasoning"])
        overall_pass = entry["tests"]["overall_pass"]
        
        total_notes = entry["tests"]["key_results"]["total"]
        diatonic_count = entry["tests"]["key_results"]["correct"]
        non_diatonic_count = entry["tests"]["key_results"]["incorrect"]
        
        key_performance[entry["prompt"]["root"]][entry["prompt"]["scale"]]["pass"] += int(overall_pass)
        key_performance[entry["prompt"]["root"]][entry["prompt"]["scale"]]["total"] += 1

        model_pass_counts[model] += int(overall_pass)
        model_total_counts[model] += 1
        
        scale_accuracy[model]["total_notes"] += total_notes
        scale_accuracy[model]["diatonic"] += diatonic_count
        scale_accuracy[model]["non_diatonic"] += non_diatonic_count
    
    return model_pass_counts, model_total_counts, scale_accuracy, key_performance

def create_bar_chart(x_data, y_data, title, xlabel, ylabel, annotate=False , figsize=(12, 6), color="skyblue"):
    """Create a standardized bar chart"""
    plt.figure(figsize=figsize)
    bars = plt.bar(x_data, y_data, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    # Annontate is to add value labels on bars
    if annotate:
        for bar in bars:
            height = bar.get_height()
            plt.annotate(
                f'{height:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8
            )
    plt.tight_layout()
    plt.show()

def create_grouped_bar_chart(root_scale_data, title="Unknown Title", annotate=True):
    """Create a grouped bar chart showing Major/Minor for each root"""
    roots = sorted(root_scale_data.keys())
    major_rates = []
    minor_rates = []
    
    for root in roots:
        # Calculate pass rates for Major and Minor
        major_total = root_scale_data[root]["Major"]["total"]
        major_pass = root_scale_data[root]["Major"]["pass"]
        major_rate = (major_pass / major_total) * 100 if major_total > 0 else 0
        
        minor_total = root_scale_data[root]["Minor"]["total"]
        minor_pass = root_scale_data[root]["Minor"]["pass"]
        minor_rate = (minor_pass / minor_total) * 100 if minor_total > 0 else 0
        
        major_rates.append(major_rate)
        minor_rates.append(minor_rate)
    
    # Create grouped bar chart
    x = np.arange(len(roots))
    width = 0.35

    plt.figure(figsize=(12, 6))
    bars1 = plt.bar(x - width/2, major_rates, width, label='Major', color='lightblue')
    bars2 = plt.bar(x + width/2, minor_rates, width, label='Minor', color='lightcoral')
    plt.xlabel('Root Note')
    plt.ylabel('Pass Rate (%)')
    plt.title(title)
    plt.xticks(x, roots)
    plt.legend()
    plt.ylim(0, 105)
    if annotate:
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                plt.annotate(
                    f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8
                )
    
    plt.tight_layout()
    plt.show()

def main():
    # Load data
    eval_log = load_data()
    
    # Basic performance analysis
    pass_counts, total_counts, scale_acc, key_perf = analyze_basic_performance(eval_log)
    models = list(total_counts.keys())

    # Overall pass percentages
    pass_percentages = [(pass_counts[m] / total_counts[m]) * 100 for m in models]
    sorted_data = sorted(zip(pass_percentages, models), reverse=True)
    pass_percentages, models = zip(*sorted_data)
    create_bar_chart(models, pass_percentages, "Overall Pass Rate by Model", "Models", "Pass Rate (%)")
    
    # Non-diatonic accuracy
    non_diatonic_pct = [(scale_acc[m]["non_diatonic"] / scale_acc[m]["total_notes"]) * 100 for m in models]
    nd_sorted_data = sorted(zip(non_diatonic_pct, models), reverse=False)
    non_diatonic_pct, models = zip(*nd_sorted_data)
    create_bar_chart(models, non_diatonic_pct, "Non-Diatonic Notes by Model", "Models", "Non-Diatonic Notes (%)")

    # Overall pass rates by key
    create_grouped_bar_chart(key_perf, title="Pass Rates by Key")

if __name__ == "__main__":
    main()