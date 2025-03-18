import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams['font.family'] = 'Times New Roman'
model = 'rf'

results_best_5years = pd.read_csv(f"results_vars/{model}_vars_comp_best_5years.csv").values * 100
results_mid_5years = pd.read_csv(f"results_vars/{model}_vars_comp_mid_5years.csv").values * 100
results_worst_5years = pd.read_csv(f"results_vars/{model}_vars_comp_worst_5years.csv").values * 100
results_best_12years = pd.read_csv(f"results_vars/{model}_vars_comp_best_12years.csv").values * 100
results_mid_12years = pd.read_csv(f"results_vars/{model}_vars_comp_mid_12years.csv").values * 100
results_worst_12years = pd.read_csv(f"results_vars/{model}_vars_comp_worst_12years.csv").values * 100

scores = [results_worst_5years, results_mid_5years, results_best_5years, 
          results_worst_12years, results_mid_12years, results_best_12years]

medians = [np.median(run) for run in scores]
bests = [np.max(run) for run in scores]
best_diff = [b - m for b, m in zip(bests, medians)]

base_colors = ['#FF6F61', '#9B59B6', '#577590']  
lighter_shades = ['#FFB6A6', '#C39BD3', '#A3C3D9']

group_colors = base_colors * 2
lighter_group_colors = lighter_shades * 2

x = np.linspace(0, 5, 6)
bar_width = 0.5

fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(x, medians, width=bar_width, color=group_colors, label="Median Accuracy")
bars2 = ax.bar(x, best_diff, width=bar_width, bottom=medians, color=lighter_group_colors, alpha=0.8, label="Best Score - Median")

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#71797E', label="Median fold accuracy"),
    Patch(facecolor='#D3D3D3', label="Best fold accuracy")
]
ax.legend(handles=legend_elements, loc="upper left", fontsize=10)

x_labels = ["Worst variables", "Mid variables", "Best variables"] * 2
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=10)

ax.text(np.mean(x[:3]), -8, "5 years", fontsize=14, ha='center')
ax.text(np.mean(x[3:]), -8, "12 years", fontsize=14, ha='center')

ax.set_ylabel("Accuracy (%)", fontsize=14)
ax.set_ylim(0, 110)

# Add value labels on top of the bars
for bar1, bar2, median, best in zip(bars1, bars2, medians, bests):
    ax.text(bar1.get_x() + bar1.get_width() / 2, bar1.get_height() / 2, f"{median:.1f}%", ha='center', va='center', color='white', fontsize=10)
    ax.text(bar2.get_x() + bar2.get_width() / 2, bar1.get_height() + bar2.get_height() - 2, f"{best:.1f}%", ha='center', va='bottom', color='black', fontsize=10)

fig.tight_layout() 
plt.show()
