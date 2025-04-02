import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams['font.family'] = 'Times New Roman'

results_best_5years_dr = pd.read_csv("results_vars/dr_vars_comp_best_5years.csv").values * 100
results_mid_5years_dr = pd.read_csv("results_vars/dr_vars_comp_mid_5years.csv").values * 100
results_worst_5years_dr = pd.read_csv("results_vars/dr_vars_comp_worst_5years.csv").values * 100
results_best_12years_dr = pd.read_csv("results_vars/dr_vars_comp_best_12years.csv").values * 100
results_mid_12years_dr = pd.read_csv("results_vars/dr_vars_comp_mid_12years.csv").values * 100
results_worst_12years_dr = pd.read_csv("results_vars/dr_vars_comp_worst_12years.csv").values * 100

results_best_5years_gru = pd.read_csv("results_vars/gru_vars_comp_best_5years_normalized.csv").values * 100
results_mid_5years_gru = pd.read_csv("results_vars/gru_vars_comp_mid_5years_normalized.csv").values * 100
results_worst_5years_gru = pd.read_csv("results_vars/gru_vars_comp_worst_5years_normalized.csv").values * 100
results_best_12years_gru = pd.read_csv("results_vars/gru_vars_comp_best_12years_normalized.csv").values * 100
results_mid_12years_gru = pd.read_csv("results_vars/gru_vars_comp_mid_12years_normalized.csv").values * 100
results_worst_12years_gru = pd.read_csv("results_vars/gru_vars_comp_worst_12years_normalized.csv").values * 100

results_best_5years_rf = pd.read_csv("results_vars/rf_vars_comp_best_5years.csv").values * 100
results_mid_5years_rf = pd.read_csv("results_vars/rf_vars_comp_mid_5years.csv").values * 100
results_worst_5years_rf = pd.read_csv("results_vars/rf_vars_comp_worst_5years.csv").values * 100
results_best_12years_rf = pd.read_csv("results_vars/rf_vars_comp_best_12years.csv").values * 100
results_mid_12years_rf = pd.read_csv("results_vars/rf_vars_comp_mid_12years.csv").values * 100
results_worst_12years_rf = pd.read_csv("results_vars/rf_vars_comp_worst_12years.csv").values * 100

scores = [results_worst_5years_dr, results_mid_5years_dr, results_best_5years_dr, 
          results_worst_12years_dr, results_mid_12years_dr, results_best_12years_dr,
          results_worst_5years_gru, results_mid_5years_gru, results_best_5years_gru, 
          results_worst_12years_gru, results_mid_12years_gru, results_best_12years_gru,
          results_worst_5years_rf, results_mid_5years_rf, results_best_5years_rf, 
          results_worst_12years_rf, results_mid_12years_rf, results_best_12years_rf]

medians = [np.median(run) for run in scores]
bests = [np.max(run) for run in scores]
best_diff = [b - m for b, m in zip(bests, medians)]

avg_median_diff = np.mean([medians[i + 2] - medians[i] for i in range(0, len(medians), 3)])
print(f"Average median difference between best and worst ranked variables: {avg_median_diff:.2f}%")

base_colors = ['#FF6F61', '#9B59B6', '#577590']
lighter_shades = ['#FFB6A6', '#C39BD3', '#A3C3D9']

group_colors = base_colors * 6
lighter_group_colors = lighter_shades * 6

x_positions = np.array([0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22])
bar_width = 0.75

fig, ax = plt.subplots(figsize=(11, 5))

bars1 = ax.bar(x_positions, medians, width=bar_width, color=group_colors, label="Median Accuracy")
bars2 = ax.bar(x_positions, best_diff, width=bar_width, bottom=medians, color=lighter_group_colors, alpha=0.8, label="Best Score - Median")

legend_elements = [
    Patch(facecolor='#71797E', label="Median fold accuracy"),
    Patch(facecolor='#D3D3D3', label="Best fold accuracy")
]
ax.legend(handles=legend_elements, loc="upper left", fontsize=10)

x_labels = ["Worst", "Mid", "Best"] * 6
ax.set_xticks(x_positions)
ax.set_xticklabels(x_labels, fontsize=12, rotation=45)

for i, label in zip([1, 5, 9, 13, 17, 21], ["5 years", "12 years", "5 years", "12 years", "5 years", "12 years"]):
    ax.text(i, -15, label, fontsize=12, ha='center', fontweight='bold')

model_positions = [3, 11, 19]  # Midpoints of each model group
model_labels = ["Detach-Rocket", "GRU", "Random Forest"]
for i, label in zip(model_positions, model_labels):
    ax.text(i, -20, label, fontsize=14, ha='center', fontweight='bold')

ax.set_ylabel("Accuracy (%)", fontsize=14)
ax.set_ylim(0, 110)

for bar1, bar2, median, best in zip(bars1, bars2, medians, bests):
    ax.text(bar1.get_x() + bar1.get_width() / 2, bar1.get_height() / 2, f"{median:.1f}%", ha='center', va='center', color='black', fontsize=11)
    ax.text(bar2.get_x() + bar2.get_width() / 2, bar1.get_height() + bar2.get_height() - 2, f"{best:.1f}%", ha='center', va='bottom', color='black', fontsize=11)

fig.tight_layout()
plt.show()
