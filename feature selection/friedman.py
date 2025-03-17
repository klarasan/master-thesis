from scipy.stats import friedmanchisquare
import numpy as np
import pandas as pd

model = 'rf'
results = pd.read_csv(f"results_vars/{model}_num_vars.csv").values * 100

print('Five years of data: ')
scores_low = results[0]   # 1 variables
scores_best = results[3]   # 4 variables
scores_high = results[7] # 8 variables

stat, p_value = friedmanchisquare(scores_low, scores_best, scores_high)

print(f"Friedman Test Statistic: {stat:.4f}")
print(f"P-value: {p_value:.4f}")

alpha = 0.05
if p_value < alpha:
    print("There is a statistically significant difference between the variable sets.")
else:
    print("No significant difference between the variable sets.")

print('Twelve years of data: ')
scores_low = results[0 + 14]   # 1 variable
scores_best = results[3 + 14]   # 4 variables 
scores_high = results[7 + 14]   # 8 variables

stat, p_value = friedmanchisquare(scores_low, scores_best, scores_high)

print(f"Friedman Test Statistic: {stat:.4f}")
print(f"P-value: {p_value:.4f}")

alpha = 0.05
if p_value < alpha:
    print("There is a statistically significant difference between the variable sets.")
else:
    print("No significant difference between the variable sets.")