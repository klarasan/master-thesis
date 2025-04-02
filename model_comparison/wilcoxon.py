import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

dr_results = pd.read_csv("results_model_comp/dr.csv")
gru_results = pd.read_csv("results_model_comp/gru_bigger_net.csv")
rf_results = pd.read_csv("results_model_comp/rf.csv")

scores_dr = dr_results.values.flatten() * 100
scores_gru = gru_results.values.flatten() * 100
scores_rf = rf_results.values.flatten() * 100

print(f"Detach-Rocket mean performance: {scores_dr.mean():.2f}")
print(f"Detach-Rocket median performance: {np.median(scores_dr):.2f}")
print(f"Detach-Rocket best performance: {np.max(scores_dr):.2f}")
print(f"Detach-Rocket worst performance: {np.min(scores_dr):.2f}")
print(f"Detach-Rocket standard deviation: {np.std(scores_dr):.2f}\n")

print(f"GRU mean performance: {scores_gru.mean():.2f}")
print(f"GRU median performance: {np.median(scores_gru):.2f}")
print(f"GRU best performance: {np.max(scores_gru):.2f}")
print(f"GRU worst performance: {np.min(scores_gru):.2f}")
print(f"GRU standard deviation: {np.std(scores_gru):.2f}\n")

print(f"Random Forest mean performance: {scores_rf.mean():.2f}")
print(f"Random Forest median performance: {np.median(scores_rf):.2f}")
print(f"Random Forest best performance: {np.max(scores_rf):.2f}")
print(f"Random Forest worst performance: {np.min(scores_rf):.2f}")
print(f"Random Forest standard deviation: {np.std(scores_rf):.2f}\n")

stat, p_value = wilcoxon(scores_dr, scores_rf, alternative='greater')

diffs = scores_dr - scores_rf
mean_diff = np.mean(diffs)
std_diff = np.std(diffs, ddof=1)
cohen_d = mean_diff / std_diff

def interpret_cohens_d(d):
    if abs(d) < 0.2:
        return "Negligible"
    elif abs(d) < 0.5:
        return "Small"
    elif abs(d) < 0.8:
        return "Medium"
    else:
        return "Large"

effect_size_interpretation = interpret_cohens_d(cohen_d)

print(f"Detach-ROCKET vs Random Forest:")
print(f"- Cohen's d: {cohen_d:.3f} ({effect_size_interpretation} effect size)")
print(f"- Wilcoxon statistic: {stat}")
print(f"- P-value: {p_value:.5f}")