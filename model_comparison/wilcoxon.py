import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

dr_results = pd.read_csv("results_model_comp/dr.csv")
gru_results = pd.read_csv("results_model_comp/gru_bigger_net.csv")
# rf_results = pd.read_csv("results_model_comp/rf.csv")

scores_dr = dr_results.values.flatten() * 100
scores_gru = gru_results.values.flatten() * 100
# scores_rf = rf_results.values.flatten() * 100

print(f"Detach-Rocket mean performance: {scores_dr.mean():.2f}")
print(f"Detach-Rocket median performance: {np.median(scores_dr):.2f}")
print(f"Detach-Rocket best performance: {np.max(scores_dr):.2f}")
print(f"Detach-Rocket worst performance: {np.min(scores_dr):.2f}")
print(f"Detach-Rocket standard deviation: {np.std(scores_dr):.2f}\n")

print(f"GRU mean performance: {scores_gru.mean():.2f}")
print(f"GRU median performance: {np.median(scores_gru):.2f}")
print(f"GRU best performance: {np.max(scores_gru):.2f}")
print(f"GRU worst performance: {np.min(scores_gru):.2f}")
print(f"GRU standard deviation: {np.std(scores_gru):.2f}")

# print(f"Random Forest mean performance: {scores_rf.mean():.2f}")
# print(f"Random Forest median performance: {np.median(scores_rf):.2f}")

# stat, p_value = wilcoxon(scores_DR, scores_GRU, alternative='greater')

# print(f"Wilcoxon statistic: {stat}")
# print(f"P-value: {p_value}")

# mean_a, mean_b = np.mean(scores_DR), np.mean(scores_GRU)
# std_a, std_b = np.std(scores_DR, ddof=1), np.std(scores_GRU, ddof=1)
# pooled_std = np.sqrt(((len(scores_DR) - 1) * std_a**2 + (len(scores_GRU) - 1) * std_b**2) / (len(scores_DR) + len(scores_GRU) - 2))
# d_value = (mean_a - mean_b) / pooled_std

# print(f"Cohen’s d: {d_value:.4f}")

# if abs(d_value) < 0.2:
#     effect = "negligible"
# elif abs(d_value) < 0.5:
#     effect = "small"
# elif abs(d_value) < 0.8:
#     effect = "medium"
# else:
#     effect = "large"
    
# print(f"Effect Size Interpretation: {effect}")