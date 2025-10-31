import json
import matplotlib.pyplot as plt
import numpy as np

# Read in the evaluation log
with open("evaluation_log.json", "r") as f:
    eval_log = json.load(f)

# Extract models and their overall_pass scores
model_pass_counts = {}
model_total_counts = {}
scale_accuracy = {}

for entry in eval_log:
    model = entry["model"]
    reasoning = entry["reasoning"]
    overall_pass = entry["tests"]["overall_pass"]

    total_notes = entry["tests"]["key_results"]["total"]
    diatonic_count = entry["tests"]["key_results"]["correct"]
    non_diatonic_count = entry["tests"]["key_results"]["incorrect"]

    # Standardizing the model names and adding reasoning info
    date = model.split("-")[-1]
    if date.isdigit():
        model = model[: -(len(date) + 1)]
    if reasoning == "True":
        model += " w/ reasoning"

    if model not in model_pass_counts:
        model_pass_counts[model] = 0
        model_total_counts[model] = 0
        scale_accuracy[model] = {"total_notes": 0, "diatonic": 0, "non_diatonic": 0}

    model_pass_counts[model] += int(overall_pass)
    model_total_counts[model] += 1

    scale_accuracy[model]["total_notes"] += total_notes
    scale_accuracy[model]["diatonic"] += diatonic_count
    scale_accuracy[model]["non_diatonic"] += non_diatonic_count

# Calculate overal pass percentages
models = list(model_pass_counts.keys())
overall_pass_percentages = [
    (model_pass_counts[model] / model_total_counts[model]) * 100 for model in models
]

# Sort models and percentages together
sorted_data = sorted(zip(overall_pass_percentages, models), reverse=True)
overall_pass_percentages, models = zip(*sorted_data) 

# Create the bar graph
plt.figure(figsize=(10, 6))
plt.bar(models, overall_pass_percentages, color="skyblue")
plt.xlabel("Models")
plt.ylabel("Percentage of Overall Pass")
plt.title("Percentage of Overall Pass Scores by Model")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

# Show the plot
plt.show()

# Create scale accuracy bar graph
accuracy_models = list(scale_accuracy.keys())
non_diatonic_accuracies = [(scale_accuracy[model]["non_diatonic"] / scale_accuracy[model]["total_notes"]) * 100 for model in models]

# Sort models and non-diatonic accuracies together
sorted_non_diatonic = sorted(zip(non_diatonic_accuracies, accuracy_models), reverse=True)
non_diatonic_accuracies, accuracy_models = zip(*sorted_non_diatonic)

# Create the bar graph
plt.figure(figsize=(10, 6))
plt.bar(accuracy_models, non_diatonic_accuracies, color="skyblue")
plt.xlabel("Models")
plt.ylabel("Percentage of Non-Diatonic Generated Notes")
plt.title("Percentage of Non-Diatonic Accuracy by Model")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

# Show the plot
plt.show()