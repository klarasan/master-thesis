import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'

def generate_seasonality_fig():
    # latitude ranges from -90 to 90 with 0 at equator
    coral = (1,0.5,0.31,0.7)
    orange = (1, 0.44, 0, 1)
    thistle = (0.9,0.75,0.9,1)
    plum = (0.5, 0, 0.5, 1)
    

    df = pd.read_csv('data/complete_dataset_12_years.csv', on_bad_lines='skip')
    north = df[(df['Latitude'] >= 38) & (df['label'] == 1)]
    mid_north = df[(df['Latitude'] >= 0) & (df['Latitude'] < 38) & (df['label'] == 1)]
    mid_south = df[(df['Latitude'] >= -33) & (df['Latitude'] < 0) & (df['label'] == 1)]
    south = df[(df['Latitude'] < -33) & df['label'] == 1]

    n_tmax, mn_tmax, ms_tmax, s_tmax = [], [], [], []
    n_srad, mn_srad, ms_srad, s_srad = [], [], [], []
    n_ppt, mn_ppt, ms_ppt, s_ppt = [], [], [], []
    for year in range(-11, -8):
        for month in range(1, 13):      
            n_tmax.append(north[f'tmax_year{year}_month{month}'].mean())
            mn_tmax.append(mid_north[f'tmax_year{year}_month{month}'].mean())
            ms_tmax.append(mid_south[f'tmax_year{year}_month{month}'].mean())
            s_tmax.append(south[f'tmax_year{year}_month{month}'].mean())

            n_srad.append(north[f'srad_year{year}_month{month}'].mean())
            mn_srad.append(mid_north[f'srad_year{year}_month{month}'].mean())
            ms_srad.append(mid_south[f'srad_year{year}_month{month}'].mean())
            s_srad.append(south[f'srad_year{year}_month{month}'].mean())

            n_ppt.append(north[f'ppt_year{year}_month{month}'].mean())
            mn_ppt.append(mid_north[f'ppt_year{year}_month{month}'].mean())
            ms_ppt.append(mid_south[f'ppt_year{year}_month{month}'].mean())
            s_ppt.append(south[f'ppt_year{year}_month{month}'].mean())

    fig, axs = plt.subplots(3, 1, figsize=(11, 9))
    axs[0].plot(range(1,37), n_tmax, label="northern most points", color=orange)
    axs[0].plot(range(1,37), mn_tmax, label="mid-northern most points", color=coral)
    axs[0].plot(range(1,37), ms_tmax, label="mid-southern most points", color=thistle)
    axs[0].plot(range(1,37), s_tmax, label="southern most points", color=plum)
    axs[0].legend(loc='lower left')
    axs[0].set_ylabel('Temperature (degrees Celsius)')
    #axs[0].set_xlabel('Months')
    #axs[0].set_title("Max temperature")

    axs[1].plot(range(1,37), n_srad, label="northern most points", color=orange)
    axs[1].plot(range(1,37), mn_srad, label="mid-northern most points", color=coral)
    axs[1].plot(range(1,37), ms_srad, label="mid-southern most points", color=thistle)
    axs[1].plot(range(1,37), s_srad, label="southern most points", color=plum)
    axs[1].legend(loc='lower left')
    axs[1].set_ylabel('Sun radiation (W/m2)')
    #axs[1].set_xlabel('Months')
    #axs[1].set_title("Sun exposure")

    axs[2].plot(range(1,37), n_ppt, label="northern most points", color=orange)
    axs[2].plot(range(1,37), mn_ppt, label="mid-northern most points", color=coral)
    axs[2].plot(range(1,37), ms_ppt, label="mid-southern most points", color=thistle)
    axs[2].plot(range(1,37), s_ppt, label="southern most points", color=plum)
    axs[2].legend(loc='lower left')
    axs[2].set_ylabel('Precipitation (mm)')
    #axs[2].set_xlabel('Months')
    #axs[2].set_title("Rain")

    fig.suptitle("Seasonality over 3 years", fontsize=13)
    #plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the main title
    plt.show()

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
    generate_seasonality_fig()