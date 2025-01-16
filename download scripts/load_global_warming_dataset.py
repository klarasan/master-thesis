import pandas as pd
import numpy as np
import random
import netCDF4 as nc
import threading
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from datetime import datetime, timedelta
from tqdm import tqdm

VARS = ['tmax', 'soil', 'ppt', 'def']

thread_local = threading.local()

def terraclimate_fetch_all(points):
    """ 
    Dataset load function optimized for fetching data for multiple points and years.
    This function minimizes the number of requests to the server.

    Parameters:
    - points: list of tuples, each containing latitude, longitude.
    
    Returns:
    - A Pandas DataFrame with climate data for each point.
    """
    all_data = []
    with ThreadPoolExecutor(max_workers=14) as executor:
        futures = []
        for var in VARS:
            data_future = executor.submit(fetch_variable, var, points)
            futures.append(data_future)
        
        for future in concurrent.futures.as_completed(futures):
            all_data.extend(future.result())

    df = pd.DataFrame(all_data)

    # Group the data by year and calculate the yearly average for each point and variable
    df_yearly = df.groupby(['Latitude', 'Longitude', 'Year', 'Variable'])['Value'].mean().reset_index()

    # Pivot the data so each variable has its own column
    df_pivot = df_yearly.pivot_table(index=['Latitude', 'Longitude', 'Year'], columns='Variable', values='Value').reset_index()

    return df_pivot

def fetch_variable(var, points):
    all_data = []
    # Fetch remote dataset
    base_url = f'http://thredds.northwestknowledge.net:8080/thredds/dodsC/agg_terraclimate_{var}_1958_CurrentYear_GLOBE.nc'
    error = True
    j = 0
    print(f"getting variable {var}")
    while(error and j < 10):
        try: 
            dataset = nc.Dataset(base_url)

            # Get latitude and longitude arrays from the dataset
            lat = dataset.variables['lat'][:]
            lon = dataset.variables['lon'][:]

            # Get time variable (days since 1900-01-01) and convert to datetime objects
            time = dataset.variables['time'][:]
            time_dates = [datetime(1900, 1, 1) + timedelta(days=t) for t in time]

            # Loop through all points and fetch data
            for point in tqdm(points, desc="Downloading point data"):
                lat_point, lon_point = point
                start_year = 1958
                event_year = 2023

                # Find the closest indices for the given latitude and longitude
                lat_idx = np.abs(lat - lat_point).argmin()
                lon_idx = np.abs(lon - lon_point).argmin()
                
                # Select time range based on the event year and number of years back
                start_date = datetime(start_year, 1, 1)
                end_date = datetime(event_year, 12, 31)
                time_indices = np.where((np.array(time_dates) >= start_date) & (np.array(time_dates) <= end_date))[0]
                times = np.array(time_dates)[time_indices]

                # Extract the data for this variable at the specified location and time
                var_data = dataset.variables[var][time_indices, lat_idx, lon_idx]

                # Append the data to a list
                for i, time in enumerate(times):
                    all_data.append({
                        'Latitude': lat_point,
                        'Longitude': lon_point,
                        'Year': time.year,
                        'Month': time.month,
                        'Variable': var,
                        'Value': var_data[i]
                    })

            dataset.close()
            error = False
        except Exception as e:
            print(f"An error occurred: {e}")
            print(f"Retrying for the {j}th time")
            j += 1
            error = True
    print(f"got variable {var}!")
    return all_data

def generate_data(points):
    """
    Generate dataset by fetching TerraClimate data for positive and negative samples.

    Returns:
    - X: Feature matrix with coordinates and climate data.
    """
    num_samples = len(points)
    num_years = 2023-1958+1
    
    X = np.zeros((num_samples, len(VARS) * num_years + 2))

    # Fetch TerraClimate data for all points
    climate_data = terraclimate_fetch_all(points)
   
    # Fill the feature matrix X
    for i, point in enumerate(points):
        lat, lon = point
        point_data = climate_data[(climate_data['Latitude'] == lat) & 
                                (climate_data['Longitude'] == lon)]
        
        if len(point_data) != num_years:
            print(f"Warning: Incomplete data for point at ({lat}, {lon})")
            continue

        # Add latitude and longitude at the beginning of the row
        X[i, 0] = lat
        X[i, 1] = lon

        # Flatten the data for each variable over the years and add to X
        for j, var in enumerate(VARS):
            X[i, 2 + j * num_years:2 + (j + 1) * num_years] = point_data[var].values

    return X


def load_tree_mortality_db(data_frac=1.0):
    """ Load the Global Tree Mortality database """
    df = pd.read_csv("data/gtm_db.csv")
    df.drop(columns=["species", "doi", "event.start"], axis=1, inplace=True)
    df = df.drop_duplicates(subset='Ref_ID')
    if data_frac < 1.0 and data_frac > 0.0:
        df = df.sample(frac=data_frac)
    return df

def extract_points(tree_mortality_df):
    points = []
    for _, row in tree_mortality_df.iterrows():
        points.append((row['lat'], row['long']))
    return points

def save_to_csv(X, path=""):
    column_names = ['Latitude', 'Longitude']  # Add coordinate columns
    for var in VARS:
        for year_offset in range(1958, 2024):
            column_names.append(f'{var}_year{year_offset}')

    df = pd.DataFrame(X, columns=column_names)

    csv_path = f'dataset_timeframe.csv'
    if path:
        csv_path = path
    df.to_csv(csv_path, index=False)

def process_data(data_frac=1.0, save_csv=True):
    """ Generate a dataset containing TerraClimate features with labels for tree mortality events. """
    tree_mortality_df = load_tree_mortality_db(data_frac)
    points = extract_points(tree_mortality_df)
    X = generate_data(points)

    if save_csv:
        save_to_csv(X)

    return X

if __name__ == "__main__":
    X = process_data()
    print(f"Data saved successfully!")