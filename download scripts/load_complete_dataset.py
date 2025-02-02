import math
import pandas as pd
import numpy as np
import random
import netCDF4 as nc
from datetime import datetime, timedelta
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import json

VARS = ['aet', 'def', 'pet', 'ppt', 'q', 'soil', 'srad', 'swe', 'tmax', 'tmin', 'vap', 'ws', 'vpd', 'PDSI']

def terraclimate_fetch_all(points, num_years):
    """ 
    Dataset load function optimized for fetching data for multiple points and years.
    This function minimizes the number of requests to the server.

    Parameters:
    - points: list of tuples, each containing Ref_ID, latitude, longitude, and die-off event year.
    - num_years: int, number of previous years to include for each event year.
    
    Returns:
    - A Pandas DataFrame with climate data for each point.
    """
    all_data = []
    with ThreadPoolExecutor(max_workers=14) as executor:
        futures = []
        for var in VARS:
            data_future = executor.submit(fetch_variable, var, points, num_years)
            futures.append(data_future)
        
        for future in concurrent.futures.as_completed(futures):
            all_data.extend(future.result())

    df = pd.DataFrame(all_data)
    df_pivot_monthly = df.pivot_table(index=['Latitude', 'Longitude', 'Year', 'Month'], columns='Variable', values='Value').reset_index()
    df_pivot_monthly.columns.name = None

    return df_pivot_monthly

def fetch_variable(var, points, num_years):
    all_data = []
    try:
        with open(f"{var}_prev_points.json", "r") as file:
            loaded_dict = json.load(file)
        previous_points = {tuple(map(int, k.strip("()").split(", "))): np.ma.array(v) for k, v in loaded_dict.items()}
    except (FileNotFoundError, json.JSONDecodeError):
        previous_points = {}    

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
            for point in tqdm(points, desc=f"Downloading point data for {var}"):
                _, lat_point, lon_point, event_year = point
                start_year = event_year - num_years

                # Find the closest indices for the given latitude and longitude
                lat_idx = np.abs(lat - lat_point).argmin()
                lat_idx = lat_idx-1 if lat[lat_idx] <= lat_point else lat_idx
                lon_idx = np.abs(lon - lon_point).argmin()
                lon_idx = lon_idx if lon[lon_idx] <= lon_point else lon_idx - 1

                # Select time range based on the event year and number of years back
                start_date = datetime(start_year, 1, 1)
                end_date = datetime(event_year, 12, 31)
                time_indices = np.where((np.array(time_dates) >= start_date) & (np.array(time_dates) <= end_date))[0]
                times = np.array(time_dates)[time_indices]

                # Extract the data for this variable at the specified location and time
                def get_var(dataset, var, time_indices, lat_idx, lon_idx):
                    if previous_points.get((lat_idx, lon_idx)) is not None:
                        return previous_points[(lat_idx, lon_idx)]
                    else:
                        val = dataset.variables[var][time_indices, lat_idx, lon_idx]
                        previous_points[(lat_idx, lon_idx)] = val
                    return val

                var_data00 = get_var(dataset, var, time_indices, lat_idx, lon_idx).data # var_data[0][1]
                var_data01 = get_var(dataset, var, time_indices, lat_idx, lon_idx+1).data # var_data[1][1]
                var_data11 = get_var(dataset, var, time_indices, lat_idx+1, lon_idx+1).data # var_data[2][1]
                var_data10 = get_var(dataset, var, time_indices, lat_idx+1, lon_idx).data  # var_data[3][1]

                # Append the data to a list
                for i, time in enumerate(times):
                    if math.isnan(var_data00[i]):
                        print('00 is nan')
                    if math.isnan(var_data01[i]):
                        print('01 is nan')
                    if math.isnan(var_data11[i]):
                        print('11 is nan')
                    if math.isnan(var_data00[i]):
                        print('10 is nan')

                    val_top = np.interp(lat_point, [lat[lat_idx], lat[lat_idx+1]], [var_data01[i], var_data11[i]])
                    val_bottom = np.interp(lat_point, [lat[lat_idx], lat[lat_idx+1]], [var_data00[i], var_data10[i]])
                    val = np.interp(lon_point, [lon[lon_idx], lon[lon_idx+1]], [val_bottom, val_top])
                    if math.isnan(val):
                        print('val is nan')

                    all_data.append({
                        'Latitude': lat_point,
                        'Longitude': lon_point,
                        'Year': time.year,
                        'Month': time.month,
                        'Variable': var,
                        'Value': val
                    })

            dataset.close()
            error = False
        except Exception as e:
            print(f"An error occurred: {e}")
            print(f"Retrying for the {j}th time")
            j += 1
            error = True

    json_dict = {str(k): v.tolist() for k, v in previous_points.items()}
    with open(f"{var}_prev_points.json", "w") as outfile: 
        json.dump(json_dict, outfile)
    return all_data

def generate_data(positive_points, negative_points, num_years):
    """
    Generate dataset by fetching TerraClimate data for positive and negative samples.

    Parameters:
    - positive_points: list of positive samples with (Ref_ID, lat, long, event_year).
    - negative_points: list of negative samples with (Ref_ID, lat, long, random_year).
    - num_years: number of years of climate data to include prior to and including event.

    Returns:
    - X: Feature matrix with coordinates and climate data.
    - Y: Labels (1 for positive samples, 0 for negative samples).
    """
    num_samples = len(positive_points) + len(negative_points)
    
    # add 3 columns for Ref_ID, lat, lon in addition to the climate variables
    X = np.zeros((num_samples, len(VARS) * num_years * 12 + 4))
    Y = np.zeros(num_samples)

    points = positive_points + negative_points
    labels = [1] * len(positive_points) + [0] * len(negative_points)

    # Fetch TerraClimate data for all points
    climate_data = terraclimate_fetch_all(points, num_years)
   
    # Fill the feature matrix X and label array Y
    for i, point in enumerate(points):
        ref_id, lat, lon, event_year = point
        point_data = climate_data[(climate_data['Latitude'] == lat) & 
                                (climate_data['Longitude'] == lon) & 
                                (climate_data['Year'] <= event_year) & 
                                (climate_data['Year'] > event_year - num_years)]
        
        point_data = point_data.sort_values(by=['Year', 'Month'])
        
        if len(point_data) != num_years * 12:
            print(f"Warning: Incomplete data for point at ({lat}, {lon}), event year {event_year}")
            continue

        # Add Ref_ID, latitude, and longitude at the beginning of the row
        X[i, 0] = ref_id
        X[i, 1] = event_year
        X[i, 2] = lat
        X[i, 3] = lon

        # Flatten the monthly data for each variable and add it to X
        for j, var in enumerate(VARS):
            X[i, 4 + j * num_years * 12:4 + (j + 1) * num_years * 12] = point_data[var].values

        Y[i] = labels[i]

    return X, Y


def load_tree_mortality_db(data_frac=1.0):
    """
    Load the Global Tree Mortality database
    
    Parameters:
    - data_frac: Fraction of total Tree Mortality dataset to load (for testing purposes).

    Returns:
    - Pandas DataFrame containing tree mortality data.
    """
    df = pd.read_csv("data/gtm_db.csv")
    df.drop(columns=["species", "doi"], axis=1, inplace=True)
    if data_frac < 1.0 and data_frac > 0.0:
        df = df.sample(frac=data_frac)
    return df


def extract_positive_points(tree_mortality_df):
    points = []
    for _, row in tree_mortality_df.iterrows():
        year = int(row['event.start'])
        points.append((row['Ref_ID'], row['lat'], row['long'], year))
    return points


def sample_negative_data(tree_mortality_df, num_years):
    points = []
    for _, row in tree_mortality_df.iterrows():
        year = int(row['event.start'])
        max_year = year - num_years
        min_year = 1958 - 1 + num_years
        if max_year >= min_year :
            year = random.randint(min_year, max_year)
            points.append((row['Ref_ID'], row['lat'], row['long'], year))
    return points

def extract_points(tree_mortality_df, num_years):
    """ Extract positive and negative points from the tree mortality data.
        Samples negative points randomly from the years 1958 to year-num_years.
    """
    pos_points = extract_positive_points(tree_mortality_df)
    neg_points = sample_negative_data(tree_mortality_df, num_years)
    return pos_points, neg_points

def save_to_csv(X, Y, num_years, path=""):
    column_names = ['Ref_ID', 'Year', 'Latitude', 'Longitude']  # Add Ref_ID, event year, and coordinate columns
    for var in VARS:
        for year_offset in range(-num_years + 1, 1):
            for month in range(1, 12+1):
                column_names.append(f'{var}_year{year_offset}_month{month}')

    df = pd.DataFrame(X, columns=column_names)
    df['label'] = Y

    csv_path = f'complete_dataset_{num_years}_years_bilinear_interp.csv'
    if path:
        csv_path = path
    df.to_csv(csv_path, index=False)

def process_data(num_prev_years, data_frac=1.0, save_csv=True):
    """ Generate a dataset containing TerraClimate features with labels for tree mortality events. """

    tree_mortality_df = load_tree_mortality_db(data_frac)
    positive_points, negative_points = extract_points(tree_mortality_df, num_prev_years)
    X, Y = generate_data(positive_points, negative_points, num_prev_years)

    # Save to CSV
    if save_csv:
        save_to_csv(X, Y, num_prev_years)

    return X, Y

if __name__ == "__main__":
    X, Y = process_data(num_prev_years=12, data_frac=1)
    print(f"Data saved successfully!")
