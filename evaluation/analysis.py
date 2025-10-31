import json
import matplotlib.pyplot as plt
import numpy as np

# Read in the evaluation log
with open("evaluation_log.json", "r") as f:
    eval_log = json.load(f)

# Extract models and their overall_pass scores
model_pass_counts = {}
model_total_counts = {}

for entry in eval_log:
    model = entry["model"]
    overall_pass = entry["tests"]["overall_pass"]

    if model not in model_pass_counts:
        model_pass_counts[model] = 0
        model_total_counts[model] = 0

    model_pass_counts[model] += int(overall_pass)
    model_total_counts[model] += 1

# Calculate percentages
models = list(model_pass_counts.keys())
percentages = [
    (model_pass_counts[model] / model_total_counts[model]) * 100 for model in models
]

# Create the bar graph
plt.figure(figsize=(10, 6))
plt.bar(models, percentages, color="skyblue")
plt.xlabel("Models")
plt.ylabel("Percentage of Overall Pass")
plt.title("Percentage of Overall Pass Scores by Model")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

# Show the plot
plt.show()