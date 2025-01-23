from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd

plt.rcParams['font.family'] = 'Times New Roman'

def datasplit70_15_15():
    df = pd.read_csv('data/dataset_avg_std_12_years.csv', on_bad_lines='skip')
    train = df[df['Ref_ID'] < 125]
    val = df[(df['Ref_ID'] >= 125) & (df['Ref_ID'] < 144)]
    test = df[df['Ref_ID'] >= 144]

    return train, val, test

def datasplit80_10_10():
    df = pd.read_csv('data/dataset_avg_std_12_years.csv', on_bad_lines='skip')
    train = df[df['Ref_ID'] < 137]
    val = df[(df['Ref_ID'] >= 137) & (df['Ref_ID'] < 145)]
    test = df[df['Ref_ID'] >= 145]

    return train, val, test

def monthly_datasplit70_15_15():
    df = pd.read_csv('data/complete_dataset_12_years.csv', on_bad_lines='skip')
    train = df[df['Ref_ID'] < 125]
    val = df[(df['Ref_ID'] >= 125) & (df['Ref_ID'] < 144)]
    test = df[df['Ref_ID'] >= 144]

    return train, val, test

def monthly_datasplit80_10_10():
    df = pd.read_csv('data/complete_dataset_12_years.csv', on_bad_lines='skip')
    train = df[df['Ref_ID'] < 137]
    val = df[(df['Ref_ID'] >= 137) & (df['Ref_ID'] < 145)]
    test = df[df['Ref_ID'] >= 145]

    return train, val, test

def data_pos_split():
    # Total: 1297
    # Train idx 1-136: 1023 samples, 78.9%
    # Val idx 137-144: 170 samples, 13.1%
    # Test idx 145-152: 104 samples, 8.0%

    # Train idx 1-124: 883 samples, 68.1%
    # Val idx 125-143: 185 samples, 14.3%
    # Test idx 144-152: 229 samples, 17.7%
    df = pd.read_csv('data/dataset_avg_std_12_years.csv', on_bad_lines='skip')
    pos_df = df[df['label'] == 1]
    ref_ids = np.zeros(153)
    for _, row in pos_df.iterrows():
        ref_ids[int(row['Ref_ID'])] += 1

    print('All samples: ' + str(pos_df.shape[0]))
    train_df = pos_df[pos_df['Ref_ID'] < 125]
    print('Training samples: ' + str(train_df.shape[0]) + ' samples, ' + str(100.0*round(train_df.shape[0]/pos_df.shape[0], 3)) + '%')
    val_df = pos_df[(pos_df['Ref_ID'] >= 125) & (pos_df['Ref_ID'] < 144)]
    print('Validation samples: ' + str(val_df.shape[0]) + ' samples, ' + str(100.0*round(val_df.shape[0]/pos_df.shape[0], 3)) + '%')
    test_df = pos_df[pos_df['Ref_ID'] >= 144]
    print('Test samples: ' + str(test_df.shape[0]) + ' samples, ' + str(100.0*round(test_df.shape[0]/pos_df.shape[0], 3)) + '%')

    return train_df, val_df, test_df

def vis_yearly_split(train_df, val_df, test_df):
    coral = (1,0.5,0.31,1)
    green = (60/255,179/255,113/255,1)
    plum = (0.5, 0, 0.5, 1)

    train_years = train_df['Year']
    train_y = np.ones(len(train_years))

    val_years = val_df['Year']
    val_y = np.ones(len(val_years)) / 2 

    test_years = test_df['Year']
    test_y = 3 * np.ones(len(test_years)) / 2 

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.scatter(train_years, train_y, s=10, c=coral, label='Training data')
    ax.scatter(val_years, val_y, s=10, c=plum, label='Validation data')
    ax.scatter(test_years, test_y, s=10, c=green, label='Test data')
    ax.set(xlim=(1965, 2020), xticks=np.arange(1970, 2020, 5), ylim=(0, 2))
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel('Event year')
    ax.set_title("Temporal data split distribution")
    ax.legend(loc= "lower left")

    fig.tight_layout()
    plt.show()

    return

def vis_geographic_split(train_df, val_df, test_df):
    coral = (1,0.5,0.31,1)
    green = (60/255,179/255,113/255,1)
    plum = (0.5, 0, 0.5, 1)
    
    train_lons = train_df['Longitude']
    train_lats = train_df['Latitude']

    val_lons = val_df['Longitude']
    val_lats = val_df['Latitude']

    test_lons = test_df['Longitude']
    test_lats = test_df['Latitude']

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([-180, 180, -180, 180], crs=ccrs.PlateCarree()) 

    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)

    # Plot points
    set_label = True
    for lon, lat in zip(train_lons, train_lats):
        if set_label == True:
            ax.plot(lon, lat, marker='o', color=coral, label="Training data", markersize=2, transform=ccrs.PlateCarree())
            set_label = False
        else:
            ax.plot(lon, lat, marker='o', color=coral, markersize=2, transform=ccrs.PlateCarree())

    set_label = True
    for lon, lat in zip(val_lons, val_lats):
        if set_label == True:
            ax.plot(lon, lat, marker='o', color=plum, label="Validation data", markersize=2, transform=ccrs.PlateCarree())
            set_label = False
        else:
            ax.plot(lon, lat, marker='o', color=plum, markersize=2, transform=ccrs.PlateCarree())

    set_label = True
    for lon, lat in zip(test_lons, test_lats):
        if set_label == True:
             ax.plot(lon, lat, marker='o', color=green, label="Test data", markersize=2, transform=ccrs.PlateCarree())
             set_label = False
        else:
            ax.plot(lon, lat, marker='o', color=green, markersize=2, transform=ccrs.PlateCarree())

    ax.set_title("Geographic data split distribution")
    ax.legend(loc= "lower left")
    plt.show()
    return

if __name__ == "__main__":
    train_df, val_df, test_df = data_pos_split()
    # vis_yearly_split(train_df, val_df, test_df)
    # vis_geographic_split(train_df, val_df, test_df)