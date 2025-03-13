from scipy.stats import friedmanchisquare
import numpy as np
import pandas as pd

results = pd.read_csv("results_vars/rf_num_vars100.csv").values * 100

print('Five years of data: ')
scores_1_var = results[0]   # 1 variable
scores_4_var = results[2]   # 3 variables
scores_9_var = results[6]   # 7 variables
scores_14_var = results[13] # 14 variables

stat, p_value = friedmanchisquare(scores_4_var, scores_9_var, scores_14_var)

print(f"Friedman Test Statistic: {stat:.4f}")
print(f"P-value: {p_value:.4f}")

alpha = 0.05
if p_value < alpha:
    print("There is a statistically significant difference between the variable sets.")
else:
    print("No significant difference between the variable sets.")

print('Twelve years of data: ')
scores_1_var = results[0 + 14]   # 1 variable
scores_4_var = results[3 + 14]   # 4 variables 
scores_9_var = results[8 + 14]   # 9 variables
scores_14_var = results[13 + 14] # 14 variables

stat, p_value = friedmanchisquare(scores_4_var, scores_9_var, scores_14_var)

print(f"Friedman Test Statistic: {stat:.4f}")
print(f"P-value: {p_value:.4f}")

alpha = 0.05
if p_value < alpha:
    print("There is a statistically significant difference between the variable sets.")
else:
    print("No significant difference between the variable sets.")