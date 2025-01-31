import numpy as np
import pandas as pd

VARS = ['aet', 'def', 'pet', 'ppt', 'q', 'soil', 'srad', 'swe', 'tmax', 'tmin', 'vap', 'ws', 'vpd', 'PDSI']

def aggregate_data(num_years):
    all_data_df = pd.read_csv('data/complete_dataset_12_years_bilinear_interp.csv', on_bad_lines='skip')
    aggr_df = all_data_df[['Ref_ID', 'Year', 'Latitude', 'Longitude']]
    
    for var in VARS:
        for year in range(1-num_years, 1):
            temp = pd.DataFrame()
            for month in range(1, 12+1):
                temp[f'{var}_year{year}_month{month}'] = all_data_df[f'{var}_year{year}_month{month}']
            aggr_df[f'{var}_year{year}_mean'] = temp.mean(axis=1)
            aggr_df[f'{var}_year{year}_std'] = temp.std(axis=1)
    
    aggr_df['label'] = all_data_df['label']
    csv_path = f'dataset_avg_std_{num_years}_years_bilinear_interp.csv'
    aggr_df.to_csv(csv_path, index=False)

    return

if __name__ == "__main__":
    aggregate_data(num_years=12)