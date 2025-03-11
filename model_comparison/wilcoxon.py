import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

detach_rocket_results = pd.read_csv("results/dr_model_comparison25models.csv")
GRU_results = pd.read_csv("results/GRU_model_comparisonS5vars_notnorm.csv")

scores_DR = detach_rocket_results.values.flatten()
scores_GRU = GRU_results.values.flatten()

stat, p_value = wilcoxon(scores_DR, scores_GRU, alternative='greater')

print(f"Wilcoxon statistic: {stat}")
print(f"P-value: {p_value}")

print(f"Detach Rocket Mean Performance: {scores_DR.mean():.4f}")
print(f"GRU Mean Performance: {scores_GRU.mean():.4f}")

print(f"Detach Rocket Median Performance: {np.median(scores_DR):.4f}")
print(f"GRU Median Performance: {np.median(scores_GRU):.4f}")

mean_a, mean_b = np.mean(scores_DR), np.mean(scores_GRU)
std_a, std_b = np.std(scores_DR, ddof=1), np.std(scores_GRU, ddof=1)
pooled_std = np.sqrt(((len(scores_DR) - 1) * std_a**2 + (len(scores_GRU) - 1) * std_b**2) / (len(scores_DR) + len(scores_GRU) - 2))
d_value = (mean_a - mean_b) / pooled_std

print(f"Cohen’s d: {d_value:.4f}")

if abs(d_value) < 0.2:
    effect = "negligible"
elif abs(d_value) < 0.5:
    effect = "small"
elif abs(d_value) < 0.8:
    effect = "medium"
else:
    effect = "large"
    
print(f"Effect Size Interpretation: {effect}")