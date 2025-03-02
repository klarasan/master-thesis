import re
import numpy as np
import pandas as pd

VARS = ['aet', 'def', 'pet', 'ppt', 'q', 'soil', 'srad', 'swe', 'tmax', 'tmin', 'vap', 'ws', 'vpd', 'PDSI']

def aggregate_data(num_years):
    all_data_df = pd.read_csv('data/12_years_bilinear_interp_w_outliers.csv', on_bad_lines='skip')
    aggr_df = all_data_df[['Ref_ID', 'Year', 'Latitude', 'Longitude']]
    
    for var in VARS:
        for year in range(1-num_years, 1):
            temp = pd.DataFrame()
            for month in range(1, 12+1):
                temp[f'{var}_year{year}_month{month}'] = all_data_df[f'{var}_year{year}_month{month}']
            aggr_df[f'{var}_year{year}_mean'] = temp.mean(axis=1)
            aggr_df[f'{var}_year{year}_std'] = temp.std(axis=1)
    
    aggr_df['label'] = all_data_df['label']
    csv_path = f'avg_std_{num_years}_years_bilinear_interp_w_outliers.csv'
    aggr_df.to_csv(csv_path, index=False)
    return

def aggregate_normalized_data():
    df = pd.read_csv('data/12_years_bilinear_interp_w_outliers.csv', on_bad_lines='skip')
    glob_norm_df = df.copy()
    loc_norm_df = df.copy()

    for var in VARS:
        var_columns = [col for col in df.columns if re.match(f"^{var}(_year-?\d+_month\d+)$", col)]
        if var_columns:
            values = df[var_columns].values.flatten()
            mean, std = values.mean(), values.std()
            glob_norm_df[var_columns] = (glob_norm_df[var_columns] - mean) / std

            row_mean = loc_norm_df[var_columns].mean(axis=1)
            row_std = loc_norm_df[var_columns].std(axis=1).replace(0, 1)
            loc_norm_df[var_columns] = loc_norm_df[var_columns].subtract(row_mean, axis=0).divide(row_std, axis=0)

    csv_path = f'globally_normalized_12_years_bilinear_interp_w_outliers.csv'
    glob_norm_df.to_csv(csv_path, index=False)

    csv_path = f'locally_normalized_12_years_bilinear_interp_w_outliers.csv'
    loc_norm_df.to_csv(csv_path, index=False)

    glob_aggr_df = df[['Ref_ID', 'Year', 'Latitude', 'Longitude']]
    loc_aggr_df = df[['Ref_ID', 'Year', 'Latitude', 'Longitude']]

    for var in VARS:
        for year in range(1-12, 1):
            glob_temp = pd.DataFrame()
            loc_temp = pd.DataFrame()
            for month in range(1, 12+1):
                glob_temp[f'{var}_year{year}_month{month}'] = glob_norm_df[f'{var}_year{year}_month{month}']
                loc_temp[f'{var}_year{year}_month{month}'] = loc_norm_df[f'{var}_year{year}_month{month}']

            glob_aggr_df[f'{var}_year{year}_mean'] = glob_temp.mean(axis=1)
            glob_aggr_df[f'{var}_year{year}_std'] = glob_temp.std(axis=1)

            loc_aggr_df[f'{var}_year{year}_mean'] = loc_temp.mean(axis=1)
            loc_aggr_df[f'{var}_year{year}_std'] = loc_temp.std(axis=1)

    glob_aggr_df['label'] = df['label']
    loc_aggr_df['label'] = df['label']

    csv_path = f'globally_normalized_avg_std_12_years_bilinear_interp_w_outliers.csv'
    glob_aggr_df.to_csv(csv_path, index=False)

    csv_path = f'locally_normalized_avg_std_12_years_bilinear_interp_w_outliers.csv'
    loc_aggr_df.to_csv(csv_path, index=False)
    return

if __name__ == "__main__":
    # aggregate_data(num_years=12)
    aggregate_normalized_data()