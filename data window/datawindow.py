import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'

gru_results = pd.read_csv("results_datawindow/gru_datawindow.csv").values * 100

gru_mean = [np.mean(run) for run in gru_results]
gru_medians = [np.median(run) for run in gru_results]

x_labels = ['0 to -4', '-1 to -5', '-2 to -6', '-3 to -7', '-4 to -8', '-5 to -9', '-6 to -10', '-7 to -11']

plt.figure(figsize=(8, 5))
plt.plot(x_labels, gru_mean, linestyle='--', label='Gru mean', color='thistle')
plt.plot(x_labels, gru_medians, linestyle='-', label='Gru median', color='plum')

plt.axhline(y=50, color='gray', linestyle=':', linewidth=1, label='50% Accuracy')

plt.xlabel("Time Window")
plt.ylabel("Accuracy (%")
plt.ylim(0, 110)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.show()
