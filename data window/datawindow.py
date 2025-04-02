import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'

gru_results = pd.read_csv("results_datawindow/gru_datawindow.csv").values * 100
dr_results = pd.read_csv("results_datawindow/dr_datawindow.csv").values * 100
rf_results = pd.read_csv("results_datawindow/rf_datawindow.csv").values * 100

gru_mean = [np.mean(run) for run in gru_results]
gru_medians = [np.median(run) for run in gru_results]

dr_mean = [np.mean(run) for run in dr_results]
dr_medians = [np.median(run) for run in dr_results]

rf_mean = [np.mean(run) for run in rf_results]
rf_medians = [np.median(run) for run in rf_results]

x_labels = ['0 to -4', '-1 to -5', '-2 to -6', '-3 to -7', '-4 to -8', '-5 to -9', '-6 to -10', '-7 to -11']

plt.figure(figsize=(8, 6))
plt.plot(x_labels, dr_medians, linestyle='-', label='Detach-ROCKET median', color='#FF6600')
plt.plot(x_labels, gru_medians, linestyle='-', label='GRU median', color='#7A4B9D')
plt.plot(x_labels, rf_medians, linestyle='-', label='Random Forest median', color='#4682B4')

plt.axhline(y=50, color='black', linestyle=':', linewidth=1, label='50% Accuracy')

plt.plot(x_labels, dr_mean, linestyle='--', label='Detach-ROCKET mean', color='#FFCC99')
plt.plot(x_labels, gru_mean, linestyle='--', label='GRU mean', color='#E6D8FF')
plt.plot(x_labels, rf_mean, linestyle='--', label='Random Forest mean', color='#ADD8E6')

plt.legend(ncol=2)

plt.xlabel("Time Window (years before event)")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 110)
plt.grid(True, linestyle='--', alpha=0.6)

plt.show()

# from scipy.stats import friedmanchisquare

# print('Random Forest:')
# print(rf_medians)
# for i in range(0, 6):
#     print(f'Data window: {i+1}')
#     scores_stabil1 = rf_results[7]   
#     scores_stabil2 = rf_results[6] 
#     scores_early = rf_results[i] 

#     stat, p_value = friedmanchisquare(scores_stabil1, scores_stabil2, scores_early)

#     print(f"Friedman Test Statistic: {stat:.4f}")
#     print(f"P-value: {p_value:.4f}\n")