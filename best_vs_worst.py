from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.family'] = 'Times New Roman'

results_best_5years = pd.read_csv("results_vars/dr_vars_comp_best_5years.csv")
scores_best_5years = results_best_5years.values * 100

print(f"Best variables, 5 years mean: {scores_best_5years.mean():.4f}")
print(f"Best variables, 5 years median: {np.median(scores_best_5years):.4f}")
print(f"Best variables, 5 years best: {np.max(scores_best_5years):.4f}\n")

results_mid_5years = pd.read_csv("results_vars/dr_vars_comp_mid_5years.csv")
scores_mid_5years = results_mid_5years.values * 100

print(f"Mid variables, 5 years mean: {scores_mid_5years.mean():.4f}")
print(f"Mid variables, 5 years median: {np.median(scores_mid_5years):.4f}")
print(f"Mid variables, 5 years best: {np.max(scores_mid_5years):.4f}\n")

results_bad_5years = pd.read_csv("results_vars/dr_vars_comp_worst_5years.csv")
scores_bad_5years = results_bad_5years.values * 100

print(f"Worst variables, 5 years mean: {scores_bad_5years.mean():.4f}")
print(f"Worst variables, 5 years median: {np.median(scores_bad_5years):.4f}")
print(f"Worst variables, 5 years best: {np.max(scores_bad_5years):.4f}\n")

scores = [scores_bad_5years, scores_mid_5years, scores_best_5years]

medians = [np.median(run) for run in scores]
bests = [np.max(run) for run in scores]
best_diff = [b - m for b, m in zip(bests, medians)]

x_labels = ["Worst vars", "Mid vars", "Best vars"] * 2
x = np.arange(len(medians))
bar_width = 0.6

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
group_colors = colors * 2

fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(x, medians, width=bar_width, color=group_colors, label="Median Accuracy")
bars2 = ax.bar(x, best_diff, width=bar_width, bottom=medians, color='gray', alpha=0.6, label="Best Score - Median")

ax.set_xticks([1, 4])
ax.set_xticklabels(["Group 1", "Group 2"], fontsize=12)
ax.set_xlabel("Experiment Groups", fontsize=14)
ax.set_ylabel("Accuracy (%)", fontsize=14)
ax.set_ylim(0, 100)

ax.legend()

# Add value labels on top of the bars
for bar1, bar2, median, best in zip(bars1, bars2, medians, bests):
    ax.text(bar1.get_x() + bar1.get_width() / 2, bar1.get_height() / 2, f"{median:.1f}%", ha='center', va='center', color='white', fontsize=10)
    ax.text(bar2.get_x() + bar2.get_width() / 2, bar1.get_height() + bar2.get_height() - 2, f"{best:.1f}%", ha='center', va='bottom', color='black', fontsize=10)

plt.title("Model Performance Across Runs", fontsize=16)
plt.show()

