import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams['font.family'] = 'Times New Roman'

model = 'rf'
results = pd.read_csv(f"results_vars/{model}_num_vars.csv").values * 100
scores_5 = results[:14]
scores_12 = results[14:]

medians_5 = [np.median(run) for run in scores_5]
means_5 = [np.mean(run) for run in scores_5]
bests_5 = [np.max(run) for run in scores_5]

medians_12 = [np.median(run) for run in scores_12]
means_12 = [np.mean(run) for run in scores_12]
bests_12 = [np.max(run) for run in scores_12]

opt_median_number_5years = np.argmax(medians_5)
opt_mean_number_5years = np.argmax(means_5)
print(f'Best median accuracy for 5 years: {medians_5[opt_median_number_5years]}, achieved with {opt_median_number_5years+1} channels')
print(f'Best mean accuracy for 5 years: {means_5[opt_mean_number_5years]}, achieved with {opt_mean_number_5years+1} channels')

opt_median_number_12years = np.argmax(medians_12)
opt_mean_number_12years = np.argmax(means_12)
print(f'Best median accuracy for 12 years: {medians_12[opt_median_number_12years]}, achieved with {opt_median_number_12years+1} channels')
print(f'Best mean accuracy for 12 years: {means_12[opt_mean_number_12years]}, achieved with {opt_mean_number_12years+1} channels')

x_values = np.arange(1, 15)

colors_5 = ['#8E44AD', '#BB8FCE', '#D2B4DE']
colors_12 = ['#E74C3C', '#F5B7B1', '#F5A996']

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(x_values, means_5, label="5 years of data - Mean", color=colors_5[0], linewidth=2)
ax.plot(x_values, medians_5, label="5 years of data - Median", color=colors_5[1], linewidth=2)
ax.plot(x_values, bests_5, label="5 years of data - Best", color=colors_5[2], linewidth=2)

ax.plot(x_values, means_12, label="12 years of data - Mean", color=colors_12[0], linewidth=2)
ax.plot(x_values, medians_12, label="12 years of data - Median", color=colors_12[1], linewidth=2)
ax.plot(x_values, bests_12, label="12 years of data- Best", color=colors_12[2], linewidth=2)

ax.set_xlabel("Number of Input Channels", fontsize=11)
ax.set_ylabel("Accuracy (%)", fontsize=11)
ax.set_xticks(x_values)  
ax.set_ylim(0, 100)  

ax.legend(fontsize=10, loc="lower right")

ax.grid(True, linestyle="--", alpha=0.6)

plt.show()