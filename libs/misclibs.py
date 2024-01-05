import pandas as pd
import numpy as np
import os
from math import sin, cos, asin, sqrt


#%%
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate haversine distance between two points
    
    Args:
        lon1: longitude point 1
        lat1: latitude point 1
        lon2: lonitude point 2
        lat2: latitude point 2
    
    Returns:
        Calculate the great circle distance between two points 
        on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6367 * c
    return km


# Get data since July, not inbetween
def get_data_from_prev_year(aso_dat, prism_dat):
    aso_dates = aso_dat.drop_duplicates(subset=['date']).reset_index(drop=True)
    years = sorted(aso_dates['year'].unique())
    
    print(f"Starting with {len(years)} years to process.")
    
    dat_ = []
    for year_ in years:
        print(f"Processing year {year_}.")
        curr_aso = aso_dates[aso_dates['year'] == year_].reset_index(drop=True)
        curr_aso = curr_aso.sort_values('date')

        prev_dat = prism_dat[(prism_dat['year'] == year_ - 1) & (prism_dat['month'] >= 8)]
        curr_dat = prism_dat[(prism_dat['year'] == year_)]
        indat = pd.concat([prev_dat, curr_dat])

        print(f"Indat shape: {indat.shape}")

        for i in range(len(curr_aso)):
            date_ = curr_aso.loc[[i], 'date'][i]
            pindat = indat[(indat['date'] <= pd.to_datetime(date_)) & (indat['date'] > pd.to_datetime(f"{year_-1}-08-01"))]

            # pindat = pindat.assign(lat_lon = pindat['latitude'].astype(str) + "_" + pindat['longitude'].astype(str))
            pindat = pindat.assign(snow = np.where(pindat['tmean'] <= 0, np.where(pindat['ppt'] > 0, 1, 0), 0))
            # pindat = pindat[pindat['snow'] == 1]
            pindat = pindat.groupby('gridNumber'). \
                agg({'snow': np.sum, 'tmean': np.sum, 'tmax': np.sum, 
                     'tmin': np.sum, 'ppt': np.sum}). \
                reset_index()
            pindat = pindat.assign(aso_date = date_)    

            print(f"Pindat shape for iteration {i}: {pindat.shape}")
            dat_.append(pindat)

        print(f"Finished processing for year {year_}.")

    aso_prism = pd.concat(dat_)
    print("Finished processing all years.")
    return aso_prism


def get_data_between_dates(aso_dat, prism_dat):
    aso_dates = aso_dat.drop_duplicates(subset=['date']).reset_index(drop=True)
    years = sorted(aso_dates['year'].unique())
    dat_ = []
    
    for year_ in years:
        curr_aso = aso_dates[aso_dates['year'] == year_].reset_index(drop=True)
        curr_aso = curr_aso.sort_values('date')

        prev_dat = prism_dat[(prism_dat['year'] == year_ - 1) & (prism_dat['month'] >= 9)]
        curr_dat = prism_dat[(prism_dat['year'] == year_ ) & (prism_dat['month'] <= 7)]
        indat = pd.concat([prev_dat, curr_dat])

        for i in range(len(curr_aso)):
            if i == 0:
                date_ = curr_aso.loc[[i], 'date'][i]
                pindat = indat[indat['date'] <= pd.to_datetime(date_)]
            else:
                date_ = curr_aso.loc[[i], 'date'][i]
                pdate_ = curr_aso.loc[[i-1], 'date'][i-1]
                pindat = indat[(indat['date'] <= pd.to_datetime(date_)) & (indat['date'] > pd.to_datetime(pdate_))]

            pindat = pindat.assign(lat_lon = pindat['latitude'].astype(str) + "_" + pindat['longitude'].astype(str))
            pindat['snow'] = np.where(pindat['tmean'] < 0, np.where(pindat['ppt'] > 0, 1, 0), 0)
            pindat = pindat[pindat['snow'] == 1]
            pindat = pindat.groupby('lat_lon').agg({'snow': np.sum, 'tmean': np.sum, 'tmax': np.sum, 'tmin': np.sum, 'ppt': np.sum, 'gridNumber': np.mean}).reset_index().drop(columns=['lat_lon'])
            pindat = pindat.assign(aso_date = date_)    
            dat_.append(pindat)

    aso_prism = pd.concat(dat_)
    return aso_prism


def check_unique_lat_lon_by_date(dataframe, year):
    dataframe['date'] = pd.to_datetime(dataframe['date'])
    
    # Filter the DataFrame for the specified year
    year_data = dataframe[dataframe['date'].dt.year == year]
    
    # Check if all "lat_lon" combinations are unique within the specified year
    unique_date_count_per_group = year_data.groupby('date')['lat_lon'].nunique()

    is_unique = unique_date_count_per_group.sum() == len(year_data)

    return is_unique


def find_closest_grid_prism_parallel(row):
    return find_closest_grid(row['lat'], row['lon'], pdat, 'gridNumber')


def find_closest_grid_nlcd_parallel(row):
    return find_closest_grid(row['lat'], row['lon'], ndat, 'lat_lon', decimal=0.001)


def find_closest_grid_elev_grade_aspect_parallel(row):
    return find_closest_grid(row['lat'], row['lon'], edat, 'index', decimal=0.01)


def find_closest_grid(lat, lon, dat, return_column, decimal=0.1):
    min_distance = np.inf
    closest_value = None

    dat = dat[(dat['lat'] >= lat - decimal) & (dat['lat'] <= lat + decimal)]
    dat = dat[(dat['lon'] >= lon - decimal) & (dat['lon'] <= lon + decimal)]

    for idx in dat.index:
        lat_p, lon_p = dat.at[idx, 'lat'], dat.at[idx, 'lon']
        distance = haversine(lat, lon, lat_p, lon_p)

        if distance < min_distance:
            min_distance = distance
            closest_value = dat.at[idx, return_column]

    return closest_value



def setup_basin(basin_name):

    # Clean up directories
    # Remove old prism data
    os.system("rm -rfv data/prism_output/")

    # create directories
    os.system(f"mkdir data/{basin_name}")
    os.system(f"mkdir data/{basin_name}/shapefiles")
    os.system(f"mkdir data/{basin_name}/elevation")
    os.system(f"mkdir data/{basin_name}/elev_grade_aspect")
    os.system(f"mkdir data/{basin_name}/NDVI")
    os.system(f"mkdir data/{basin_name}/processed")
    os.system(f"mkdir data/{basin_name}/models")
    os.system(f"mkdir data/{basin_name}/predictions")
    os.system(f"mkdir data/prism_output/")
    os.system(f"mkdir data/prism_output/daily")
    os.system(f"mkdir data/{basin_name}/NLCD/")
    os.system(f"mkdir data/{basin_name}/NLCD/processed/")