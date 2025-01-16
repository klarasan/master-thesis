import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'

def generate_seasonality_fig():
    # latitude ranges from -90 to 90 with 0 at equator
    return

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def generate_global_warming_fig():
    df = pd.read_csv("data/dataset_timeframe.csv")
    years = range(1958, 2024)

    temp_means = []
    def_means = []
    for year in years:
        temp_means.append(df[f'tmax_year{year}'].mean())
        def_means.append(df[f'def_year{year}'].mean())

    window_size = 9  # Adjust the size as needed
    smoothed_temps = moving_average(temp_means, window_size)
    smoothed_defs = moving_average(def_means, window_size)

    fig, axs = plt.subplots(1, 2, figsize=(10, 8))

    axs[0].plot(years, temp_means, label="Average maxtemperature", color=(1,0.5,0.31,0.35))
    axs[0].plot(range(1962, 2020), smoothed_temps, label="Smoothed maxtemperature", color=(1,0.5,0.31,1))
    axs[0].set_title("Max temperature")
    axs[0].legend()

    axs[1].plot(years, def_means, label="Average water deficit", color=(1,0,0,0.35))
    axs[1].plot(range(1962, 2020), smoothed_defs, label="Smoothed water deficit", color=(1,0,0,1))
    axs[1].set_title("Climate water deficit")
    axs[1].legend()

    fig.suptitle("Global trends in dataset", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the main title
    plt.show()

    return

if __name__ == "__main__":
    generate_global_warming_fig()