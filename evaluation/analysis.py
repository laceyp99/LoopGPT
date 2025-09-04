import json
import matplotlib.pyplot as plt
import numpy as np

# Read in the evaluation log
with open("evaluation_log.json", "r") as f:
    eval_log = json.load(f)

model_data = {}
for test in eval_log:
    model = test["model"]
    if model not in model_data:
        model_data[model] = {
            "tests": {
                # "bar length": 0,
                "in key": 0,
                "note duration": 0,
                "all correct": 0
            },
            "stats": {
                "cost": 0,
                "latency": 0
            },
            "total": 0
        }

    model_data[model]["total"] += 1
    # model_data[model]["tests"]["bar length"] += 1 if test["bar_count_pass"] else 0
    model_data[model]["tests"]["in key"] += 1 if test["in_key_pass"] else 0
    model_data[model]["tests"]["note duration"] += 1 if test["note_length_pass"] else 0
    model_data[model]["tests"]["all correct"] += 1 if test["output_pass"] else 0
    model_data[model]["stats"]["cost"] += test["cost"]
    model_data[model]["stats"]["latency"] += test["api_latency"]
    
for model in model_data:
    total = model_data[model]["total"]
    for test_name in model_data[model]["tests"].keys():
        model_data[model]["tests"][test_name] = (model_data[model]["tests"][test_name] / total) * 100.0
    for stat_name in model_data[model]["stats"].keys():
        model_data[model]["stats"][stat_name] = model_data[model]["stats"][stat_name] / total

# # If "bar length" is a fraction (0-1), convert to percentage
# bar_length_values = [model["tests"]["bar length"] for model in model_data.values()]
in_key_values = [model["tests"]["in key"] for model in model_data.values()]
note_duration_values = [model["tests"]["note duration"] for model in model_data.values()]
all_correct_values = [model["tests"]["all correct"] for model in model_data.values()]

latency_values = [model["stats"]["latency"] for model in model_data.values()]
cost_values = [model["stats"]["cost"] for model in model_data.values()]

# Average test scores per Model
plt.figure(figsize=(10, 8))
plt.barh(model_data.keys(), all_correct_values, label="All Correct")
plt.title("Average Test Scores per Model")
plt.xlabel("Percentage (%)")
plt.tight_layout()
plt.savefig("evaluation/Results/average_test_scores.png")
plt.show()

# Average test scores per Model across all tests
plt.figure(figsize=(10, 8))
y = np.arange(len(model_data.keys()))
height = 0.25
plt.barh(y, in_key_values, height=height, label="In Key")
plt.barh(y - height, note_duration_values, height=height, label="Note Duration")
# plt.barh(y + height, bar_length_values, height=height, label="Bar Length")
plt.title("Test Percentages per Model")
plt.ylabel("Model")
plt.yticks(y - height / 2, model_data.keys())
plt.xlabel("Percentage (%)")
plt.legend() # loc='upper left')
plt.tight_layout()
plt.savefig("evaluation/Results/accuracy.png")
plt.show()

# Average Latency per Model
plt.figure(figsize=(10, 8))
plt.barh(model_data.keys(), latency_values, label="Avg Latency (s)")
plt.title("Average Latency per Model")
plt.xlabel("Latency (s)")
plt.tight_layout()
plt.savefig("evaluation/Results/latency.png")
plt.show()

# Average Cost per Model
plt.figure(figsize=(10, 8))
plt.barh(model_data.keys(), cost_values, label="Avg Cost ($)")
plt.title("Average Cost per Model")
plt.xlabel("Cost ($)")
plt.tight_layout()
plt.savefig("evaluation/Results/cost.png")
plt.show()