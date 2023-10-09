import os
import urllib
import requests
import urllib.request
import time
import multiprocessing
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
import xarray as xr
import rioxarray as rxr
from shapely.geometry import Point
from bs4 import BeautifulSoup

import dask.dataframe as dd
from dask import delayed
from dask.diagnostics import ProgressBar

from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

import ee

from haversine import haversine

from libs.asolibs import *

# Initialize EE service account
EE_ACCOUNT = 'gee-satellite-proc@noble-maxim-311917.iam.gserviceaccount.com'
EE_PRIVATE_KEY_FILE = 'gee-privatekey.json'
EE_CREDENTIALS = ee.ServiceAccountCredentials(EE_ACCOUNT, EE_PRIVATE_KEY_FILE)
ee.Initialize(EE_CREDENTIALS)



# Get uniuq aso lat and lon coords
def get_aso_lat_lon(gdf):
    aso = proc_ASO_SWE_shp(gdf)
    aso = aso.dropna()
    aso = aso[aso['SWE'] >= 0]
    # Get unique lat/lon coords
    print("Defining grids")
    lat_lon_dat = aso.assign(lat = np.round(aso['lat'], 4),
        lon = np.round(aso['lon'], 4))
    
    lat_lon_dat['lat_lon'] = lat_lon_dat['lat'].astype(str) + "_" + lat_lon_dat['lon'].astype(str)

    print("Grouping by and getting averages")
    lat_lon_dat = lat_lon_dat.groupby(['date', 'site', 'lat_lon']).agg({'SWE': np.mean, 'lat': np.mean, 'lon': np.mean}).reset_index()

    len(lat_lon_dat['lat_lon'].unique())

    print("Subsetting unique values")
    lat_lon_dat = lat_lon_dat[['lat', 'lon', 'lat_lon', 'SWE']]
    lat_lon_dat = lat_lon_dat.sort_values(['lat', 'lon']).reset_index(drop=True)
    return lat_lon_dat



def find_closest_grid(lat, lon, dat, return_column):
    min_distance = np.inf
    closest_value = None

    dat = dat[(dat['lat'] >= lat - 0.01) & (dat['lat'] <= lat + 0.01)]
    dat = dat[(dat['lon'] >= lon - 0.01) & (dat['lon'] <= lon + 0.01)]

    for idx in dat.index:
        lat_p, lon_p = dat.at[idx, 'lat'], dat.at[idx, 'lon']
        distance = haversine((lat, lon), (lat_p, lon_p))

        if distance < min_distance:
            min_distance = distance
            closest_value = dat.at[idx, return_column]

    return closest_value




# --------------------------------
# Load ASO data
# Load shapefile
shape_loc = "data/shapefiles/tuolumne_watershed/Tuolumne_Watershed.shp"
gage_id = "11290000"

gdf = gpd.read_file(shape_loc)
gdf = gdf.to_crs(epsg=4326)

# Get ASO Data
aso_dat = proc_ASO_SWE_shp(gdf)
aso_dat = aso_dat.dropna()
aso_dat = aso_dat[aso_dat['SWE'] >= 0]







# -----------------------------------------------------------
# Elevation data

def get_elevation(lat, lon, num_retries=3):
    url = 'https://epqs.nationalmap.gov/v1/json?'
    params = {
        'output': 'json',
        'x': lon,
        'y': lat,
        'units': 'Meters'
    }
    for i in range(num_retries):
        try:
            result = requests.get((url + urllib.parse.urlencode(params)))
            return result.json()['value']
        except Exception as e:
            print(f"Failed to retrieve elevation for coordinates ({lat}, {lon}): {str(e)}. Retry attempt: {i+1}")
            time.sleep(5)  # wait for 1 second before retrying
    print(f"Failed to retrieve elevation after {num_retries} attempts")
    return "NA"


def save_elevation(lat, lon, main_directory):
    file_name = f"{lat}_{lon}.csv"
    file_path = os.path.join(main_directory, file_name)

    # Check if the file already exists
    if os.path.isfile(file_path):
        print(f"File '{file_name}' already exists. Skipping.")
        return
    
    elevation = get_elevation(lat, lon)
    
    # Save the elevation as a separate file
    with open(file_path, 'w') as file:
        file.write(f"Latitude,Longitude,Elevation\n")
        file.write(f"{lat},{lon},{elevation}\n")


def elevation_function(df, lat_column, lon_column, main_directory, max_workers=8):
    """Query service using lat, lon. Save individual files for each coordinate."""
    
    def fetch_elevation(row):
        lat = row[1][lat_column]
        lon = row[1][lon_column]
        save_elevation(lat, lon, main_directory)
    
    # Using ThreadPoolExecutor to parallelize the requests
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Wrap the executor.map with tqdm for a progress bar
        _ = list(tqdm(executor.map(fetch_elevation, df.iterrows()), total=len(df), desc="Fetching elevation data"))



# Get elevation data for each lat/lon
elevation_function(lat_lon_dat, 'lat', 'lon', "data/Tuolumne_Watershed/elevation/", 56)

elev_files = glob.glob("data/Tuolumne_Watershed/elevation/*.csv")

df_ = []
for file_ in tqdm(elev_files):
    df = pd.read_csv(file_)
    df_.append(df)

eldat = pd.concat(df_)
eldat.to_parquet("data/Tuolumne_Watershed/aso_elevation.parquet", compression=None, index=False)






# -----------------------------------------------------
# Get slope/aspect data
# TODO





# -----------------------------------------------------
# Get NDVI

# 1. Download and clip to Tuolumne 

year = "2012"
def get_ndvi_filelist(year):

    url = f"https://www.ncei.noaa.gov/data/land-normalized-difference-vegetation-index/access/{year}/"

    # Fetch the content of the webpage
    response = requests.get(url)

    # Parse the webpage content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all the links in the webpage
    links = soup.find_all('a')

    # Print the URLs of the files
    lst_ = []
    for link in links:
        href = link.get('href')
        if href:  # Check if the href attribute is not None
            file_url = url + href
            lst_.append(file_url)
    lst_ = lst_[5:]

    # Define start and end dates as strings in the format YYYYMMDD
    start_date = f"{year}0101"
    # start_date = f"{year}0601"
    end_date = f"{year}0731"

    # Filter the list of files
    filtered_urls = []
    for file_url in lst_:
        # Extract the date string from the file name (assuming it's always in the same position)
        date_string = file_url.split('_')[-2]  # Split by underscore and get the second to last part
        
        # Check if the date_string is within the desired range
        if start_date <= date_string <= end_date:
            filtered_urls.append(file_url)

    return filtered_urls



# 2012-2014
year = "2012"
def proc_ndvi(year, gdf):
    
    files = get_ndvi_filelist(year)
    max_retry = 5  # Define max retry attempts for downloading a file.

    dat_ = []
    for url in files:
        success = False
        attempts = 0
        while not success and attempts < max_retry:
            try:
                print(f"Attempt {attempts+1}: Downloading {url}")
                # Download the netCDF file
                filename = url.split("/")[-1]
                os.system(f"wget {url} -O {filename}")

                # Try to open the netCDF file using xarray
                ds = xr.open_dataset(filename)
                success = True
            except Exception as e:
                attempts += 1
                print(f"Failed to download or open file {url}, retrying ({attempts}/{max_retry})")
                if attempts < max_retry:
                    time.sleep(60)  # Sleep for 1 minute before retrying

        if success:
            try:
                ds = xr.open_dataset(filename)
                ds = ds['NDVI']
                ds = ds.rio.write_crs("EPSG:4326")

                # Proceed with processing
                dat = ds.rio.clip(gdf.geometry, all_touched=True, drop=True, invert=False, from_disk=True)
                dat = dat.to_series().reset_index()
                outdat = dat.dropna()
                dat_.append(outdat)
                
                # Optionally delete the file after processing
                os.system(f"rm {filename}")

            except Exception as e:
                print(f"Error processing file {url}: {e}")

    if dat_:
        # Save the results to a CSV file if there's any data
        rdat = pd.concat(dat_)
        rdat.to_csv(f"data/Tuolumne_Watershed/NDVI/{year}.csv", index=False)
    else:
        print(f"No data available for year {year}")



[proc_ndvi(int(year_), gdf) for year_ in np.linspace(2014, 2023, len(range(2014, 2024)))]


[download_and_filter_netcdf(str(int(year_)), gdf) for year_ in np.linspace(2012, 2024, 13)]


ds = xr.open_dataset("NLDAS_elevation.nc4")

ds = ds['NLDAS_elev']
ds = ds.rio.write_crs("EPSG:4326")

# Proceed with processing
dat = ds.rio.clip(gdf.geometry, all_touched=True, drop=True, invert=False, from_disk=True)
dat = dat.to_series().reset_index()
dat.dropna().to_csv("test.csv", index=False)




lat = dataframe['lat'][0]
lon = dataframe['lon'][0]
date = dataframe['date'][0]

start_date = '2014-01-01'
end_date = '2023-12-31'

def get_ndvi(lat, lon, start_date, end_date):
    """
    Fetch NDVI data for a given latitude/longitude coordinate and date range.
    
    Args:
        lat (float): The latitude of the coordinate.
        lon (float): The longitude of the coordinate.
        start_date (str): The start date for fetching data in 'YYYY-MM-DD' format.
        end_date (str): The end date for fetching data in 'YYYY-MM-DD' format.
    
    Returns:
        list: A list of tuples containing date and NDVI values for each day within the date range.
    """
    def apply_rolling_mean(ndvi_values, window_size=3):
        """
        Apply rolling mean to the NDVI values.
        
        Args:
            ndvi_values (list): List of tuples containing date and NDVI values.
            window_size (int): Size of the rolling window.
        
        Returns:
            list: A list of tuples containing date and rolling mean NDVI values.
        """
        rolling_ndvi = []
        values = [val[1] for val in ndvi_values]
        for i in range(len(ndvi_values)):
            if i < window_size // 2 or i >= len(ndvi_values) - window_size // 2:
                rolling_ndvi.append((ndvi_values[i][0], "NA"))
            else:
                mean = sum(values[i - window_size // 2: i + window_size // 2 + 1]) / window_size
                rolling_ndvi.append((ndvi_values[i][0], mean))
        return rolling_ndvi

    # Define a point geometry from the coordinates
    point = ee.Geometry.Point(lon, lat)
    
    # Use the MODIS NDVI dataset
    dataset = ee.ImageCollection('MODIS/006/MOD13A2').filterDate(start_date, end_date)
    
    # Initialize an empty list to store the results
    ndvi_values = []
    
    # Iterate over each day within the date range
    date = start_date
    while date <= end_date:
        # Calculate the end date as the next day
        next_date = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        
        # Filter the dataset for the current day
        daily_dataset = dataset.filter(ee.Filter.date(date, next_date))
        
        # Calculate the average NDVI for the current day
        ndvi = daily_dataset.select('NDVI').mean()
        
        try:
            # Sample the NDVI at the point location and get the result
            result = ndvi.sample(region=point, scale=500, projection='EPSG:4326').first().get('NDVI').getInfo()
            ndvi_values.append((date, result * 0.0001))
        except Exception as e:
            print(f"Failed to retrieve NDVI for date {date}: {str(e)}")
            ndvi_values.append((date, "NA"))
        
        # Increment the date to the next day
        date = next_date
    
    # Apply rolling mean through the NDVI values
    ndvi_values = apply_rolling_mean(ndvi_values)
    
    return ndvi_values



def process_ndvi_data(dataframe):
    """
    Process a DataFrame to fetch NDVI data for each row.
    
    Args:
        dataframe (pd.DataFrame): DataFrame containing 'lat', 'lon', and 'date' columns.
    
    Returns:
        pd.DataFrame: DataFrame with an additional 'ndvi' column.
    """
    
    # Function to apply to each row of the DataFrame
    def fetch_ndvi(row):
        return get_ndvi(row[2], row[3], row[1])
    
    # Apply the fetch_ndvi function to each row and add a progress bar
    ndvi_values = [fetch_ndvi(row) for row in tqdm(dataframe.itertuples(), total=len(dataframe), desc="Fetching NDVI data")]
    
    # Add the NDVI values as a new column to the DataFrame
    dataframe['ndvi'] = ndvi_values
    
    return dataframe



# Example usage:
lat = 37.7749  # Latitude for San Francisco, CA
lon = -122.4194  # Longitude for San Francisco, CA

# Fetch NDVI data
ndvi = get_ndvi(lat, lon)
print(f'Average NDVI: {ndvi}')

ndvi_dat = aso_dat.drop_duplicates(subset=['date', 'lat_lon'])
ndvi_dat = ndvi_dat[['date', 'lat', 'lon', 'lat_lon']].reset_index(drop=True)
process_ndvi_data(ndvi_dat.iloc[[0], :])


# Bind data together
df_ = []
ndvi_files = glob.glob('data/Tuolumne_Watershed/NDVI/*.csv')
for file_ in ndvi_files:
    df = pd.read_csv(file_)
    df_.append(df)

ndat = pd.concat(df_)
ndat['lat_lon'] = ndat['latitude'].astype(str) + "_" + ndat['longitude'].astype(str)
ndat = ndat.drop_duplicates(subset='lat_lon').reset_index(drop=True)
ndat = ndat[['latitude', 'longitude']].reset_index()
ndat.columns = ['ndvi_grid', 'lat', 'lon']


# Enable progress_apply in pandas
tqdm.pandas()

ldat = get_aso_lat_lon(gdf)

# Use apply to find the closest grid for all rows
ldat['ndvi_grid'] = ldat.progress_apply(lambda row: find_closest_grid(row['lat'], row['lon'], ndat, 'ndvi_grid'), axis=1)
ldat.to_csv('data/Tuolumne_Watershed/aso_ndvi_lookup.csv', index=False)



# -------------------------------
# PRISM
pdat = pd.read_csv("data/Tuolumne_Watershed/Tuolumne_Watershed_PRISM_daily_1981-2022.csv")
pdat = pdat.drop_duplicates(subset=['gridNumber'])
pdat = pdat[['longitude', 'latitude', 'gridNumber']]
pdat.columns = ['lon', 'lat', 'gridNumber']

shape_loc = "data/shapefiles/tuolumne_watershed/Tuolumne_Watershed.shp"

gdf = gpd.read_file(shape_loc)
gdf = gdf.to_crs(epsg=4326)

ldat = get_aso_lat_lon(gdf)
ldat = ldat[['lon', 'lat']].reset_index(drop=True)
ldat = ldat.assign(lat_lon = ldat['lat'].astype(str) + "_" + ldat['lon'].astype(str))





def find_closest_grid_parallel(row):
    return find_closest_grid(row['lat'], row['lon'], pdat, 'gridNumber')

# Convert DataFrame to list of dictionaries
rows = ldat.to_dict(orient='records')

# Set the number of workers
num_workers = multiprocessing.cpu_count()

# Initialize multiprocessing Pool
with multiprocessing.Pool(num_workers) as pool:
    results = list(tqdm(pool.imap(find_closest_grid_parallel, rows), total=len(rows), desc="Processing:"))

# Assign the results to the DataFrame
ldat['prism_grid'] = results
ldat.dropna()
ldat.to_csv('data/Tuolumne_Watershed/aso_prism_lookup.csv', index=False)










lon = -119.2713
lat = 37.7396
return_column = "gridNumber"
dat = pdat

def find_closest_grid(lat, lon, dat, return_column):
    min_distance = np.inf
    closest_value = None

    dat = dat[(dat['lat'] >= lat - 0.01) & (dat['lat'] <= lat + 0.01)]
    dat = dat[(dat['lon'] >= lon - 0.01) & (dat['lon'] <= lon + 0.01)]

    for idx in dat.index:
        lat_p, lon_p = dat.at[idx, 'lat'], dat.at[idx, 'lon']
        distance = haversine((lat, lon), (lat_p, lon_p))

        if distance < min_distance:
            min_distance = distance
            closest_value = dat.at[idx, return_column]

    return closest_value

# Enable progress_apply in pandas
tqdm.pandas()

# Use apply to find the closest grid for all rows
ldat['prism_grid'] = ldat.progress_apply(lambda row: find_closest_grid(row['lat'], row['lon'], pdat, 'gridNumber'), axis=1)
ldat.to_csv('data/Tuolumne_Watershed/aso_prism_lookup.csv', index=False)



# ----------------------------------
# Land cover
nlcd_files  = ["nlcd_2019_land_cover_l48_20210604.img",
               "nlcd_2016_land_cover_l48_20210604.img",
               "nlcd_2013_land_cover_l48_20210604.img"]

def get_nlcd(filename):

    # Get year
    year = filename.split("_")[1]

    save_filename = filename.split("/")[-1].split(".")[0]

    # Open the dataset
    ds = rxr.open_rasterio(f"data/NLCD/raw/{filename}")

    # Reproject to EPSG:4326
    # ds = ds.rio.reproject("EPSG:4326")

    # Load the shapefile
    shape_loc = "data/shapefiles/tuolumne_watershed/Tuolumne_Watershed.shp"

    target_crs_wkt = 'PROJCS["Albers Conical Equal Area",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Albers_Conic_Equal_Area"],PARAMETER["latitude_of_center",23],PARAMETER["longitude_of_center",-96],PARAMETER["standard_parallel_1",29.5],PARAMETER["standard_parallel_2",45.5],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["meters",1],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
    gdf = gpd.read_file(shape_loc)
    gdf = gdf.to_crs(target_crs_wkt)
    # gdf = gdf.to_crs(epsg="4326")

    outdat = ds.rio.clip(gdf.geometry, all_touched=True, drop=True, invert=False, from_disk=True) 
    outdat = outdat.rio.reproject("EPSG:4326")
    outdat = outdat.to_series().reset_index()
    outdat.columns = ['band', 'lat', 'lon', 'landcover']
    outdat = outdat.assign(year = year)
    outdat = outdat[['year', 'lat', 'lon', 'landcover']]

    # Filter no class
    outdat = outdat[outdat['landcover'] != 0]
    outdat = outdat[outdat['landcover'] != 255]

    outdat.to_csv(f"data/NLCD/processed/{save_filename}.csv", index=False)
    print(f"Saved: data/NLCD/processed/{save_filename}.csv")


[get_nlcd(x) for x in nlcd_files]


ldat = get_aso_lat_lon(gdf)
ldat = ldat[['lon', 'lat']].reset_index(drop=True)
ldat.to_csv("data/Tuolumne_Watershed/unique_ASO_grids.csv", index=False)

ldat = pd.read_csv("data/Tuolumne_Watershed/unique_ASO_grids.csv")

ndat = pd.read_csv("data/NLCD/processed/nlcd_2019_land_cover_l48_20210604.csv")
ndat = ndat.assign(lat_lon = ndat['lat'].astype(str) + "_" + ndat['lon'].astype(str))


def find_closest_grid_parallel(row):
    return find_closest_grid(row['lat'], row['lon'], ndat, 'lat_lon')

# Convert DataFrame to list of dictionaries
rows = ldat.to_dict(orient='records')

# Set the number of workers
num_workers = multiprocessing.cpu_count()

# Initialize multiprocessing Pool
with multiprocessing.Pool(num_workers) as pool:
    results = list(tqdm(pool.imap(find_closest_grid_parallel, rows), total=len(rows), desc="Processing:"))

# Assign the results to the DataFrame
ldat['nlcd_grid'] = results
len(ldat.dropna()) == len(ldat)
ldat.to_csv('data/Tuolumne_Watershed/aso_nlcd_lookup.csv', index=False)







# -----------------------------------------------------------
# Get snotel station data
#   Currently sourced using "" and exporting as a csv to the working directory
parent_dir = '.'
sno_file = f'{parent_dir}/TuolumneSWE_041981_042023.txt'
sno_data = pd.read_csv(sno_file, comment="#")

# get columns to drop
incomplete_colms = sno_data.isna()\
    .any(axis=0)\
    .where(lambda x: x)\
    .dropna()\
    .index.tolist()

# make sure Date sticks around
try:
    assert "Date" not in incomplete_colms
except AssertionError:
    print(f"ERROR: `Date` column of {sno_file} contained NaT entries.")

# format date -> year
nrdat = nrdat.assign(year=ndat.Date.apply(
        lambda x: dt.datetime.strptime(x, "%b %Y").year
    ))\
    .drop(columns=["Date"])

# filter for continuous time series & shorten column names
nrdat = nrdat.drop(columns=incomplete_colms)\
    .rename(columns=zip(map(
        lambda x: x.replace("Snow Water Equivalent (in) Start of Month Values", "SWE (in)"), incomplete_colms)
    ))

# save processed data
nrdat.to_csv(f"data/{dir_name}/{dir_name}_NRCS_SNOTEL_data.csv", index = False)







# --------------------------------------------------------------------
# Bind data together
# -----------------------------------------------------------
# Load shapefile
shape_loc = "data/shapefiles/tuolumne_watershed/Tuolumne_Watershed.shp"
gdf = gpd.read_file(shape_loc)
gdf = gdf.to_crs(epsg=4326)

# Get ASO Data
aso_dat = proc_ASO_SWE_shp(gdf)
aso_dat = aso_dat.dropna()
aso_dat = aso_dat[aso_dat['SWE'] >= 0]

len(aso_dat)    # 28906376

aso_dat = aso_dat.assign(lat = np.round(aso_dat['lat'], 4),
                         lon = np.round(aso_dat['lon'], 4))

aso_dat['lat_lon'] = aso_dat['lat'].astype(str) + "_" + aso_dat['lon'].astype(str)


print("Grouping by and getting averages")
aso_dat = aso_dat.groupby(['date', 'lat_lon']).agg({'SWE': np.mean, 'lat': np.mean, 'lon': np.mean}).reset_index()


check_unique_lat_lon_by_date(aso_dat, 2016)

[check_unique_lat_lon_by_date(aso_dat, x) for x in [2013, 2016, 2017, 2018, 2019, 2020, 2021]]


len(aso_dat)    # 28906376

aso_dat = aso_dat.assign(year = pd.to_datetime(aso_dat['date']).dt.strftime("%Y"),
    month = pd.to_datetime(aso_dat['date']).dt.strftime("%m"))
aso_dat = aso_dat.assign(year = aso_dat['year'].astype(int), month = aso_dat['month'].astype(int))


# Get elevation data
aso_elev = pd.read_parquet("data/Tuolumne_Watershed/aso_elevation.parquet")
aso_elev.columns = ['lat', 'lon', 'elevation']
aso_elev = aso_elev.assign(lat_lon = aso_elev['lat'].astype(str) + "_" + aso_elev['lon'].astype(str))
aso_elev = aso_elev.drop(columns=['lat', 'lon'])

# Check that all values are unique
assert len(aso_elev['lat_lon'].unique()) == len(aso_elev)


# Prism data and get data inbetween dates
prism_dat = pd.read_csv("data/Tuolumne_Watershed/Tuolumne_Watershed_PRISM_daily_1981-2020.csv")
prism_dat = prism_dat.assign(date = pd.to_datetime(prism_dat['date'], format = "%Y%m%d"))
prism_dat = prism_dat.assign(month = pd.to_datetime(prism_dat['date']).dt.strftime("%m"))
prism_dat = prism_dat.assign(year = pd.to_datetime(prism_dat['date']).dt.strftime("%Y"))

prism_dat = prism_dat.assign(month = prism_dat['month'].astype(int),
                             year = prism_dat['year'].astype(int))

# prism_dat.drop_duplicates()

# prism_dat.groupby(['var']).count()

# len(prism_dat['date'].unique())


# test = prism_dat[(prism_dat['date'] == '2012-12-31') & (prism_dat['gridNumber'] == 407556)]
# test


# var           date   longitude   latitude  gridNumber  month  year     ppt    tmax       tmin    tmean
# 2882628 2012-12-31 -120.583333  37.791667      407556     12  2012     NaN  10.156   1.314000   5.7350
# 2882629 2012-12-31 -120.583333  37.791667      407556     12  2012  0.0000     NaN        NaN      NaN
# 2882630 2012-12-31 -120.541667  37.791667      407557     12  2012     NaN  10.094   1.301000   5.6975
# 2882631 2012-12-31 -120.541667  37.833333      406152     12  2012     NaN   9.985   1.266000   5.6255
# 2882632 2012-12-31 -120.541667  37.791667      407557     12  2012  0.0000     NaN        NaN      NaN

# len(test) == len(test.dropna())

prism_dat = prism_dat.pivot_table(index=['date', 'gridNumber', 'month', 'year'],
                            columns='var', aggfunc=np.nanmean,
                            values='value').reset_index()

prism_dat = prism_dat.assign(tmean = (prism_dat['tmax'] + prism_dat['tmin'])/2)

# Fix duplicates nas (manually checked on prism)

# Check for NA
indat_copy = prism_dat.copy()

# Drop rows with NaN values from the copied DataFrame
indat_copy = indat_copy.dropna()

# Find rows with NaN values by comparing the original and copied DataFrames
rows_with_na = indat[~indat.index.isin(indat_copy.index)]
rows_with_na


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

prism_dat = get_data_from_prev_year(aso_dat, prism_dat)

len(prism_dat) == len(prism_dat.dropna())

prism_dat.groupby('aso_date').count().sum() == len(prism_dat['gridNumber'].unique())*len(prism_dat['aso_date'].unique())


# Get data in between aso dates
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



# Get data inbetween dates
# aso_dates = aso_dat.drop_duplicates(subset=['date']).reset_index(drop=True)
# years = sorted(aso_dates['year'].unique())


# # year_ = 2013
# # for loop through each year
# dat_ = []
# for year_ in years:
#     curr_aso = aso_dates[aso_dates['year'] == year_].reset_index(drop=True)
#     curr_aso = curr_aso.sort_values('date')

#     # Get current year and previous year of prism
#     prev_dat = prism_dat[(prism_dat['year'] == year_ - 1) & (prism_dat['month'] >= 9)]
#     curr_dat = prism_dat[(prism_dat['year'] == year_ ) & (prism_dat['month'] <= 7)]
#     indat = pd.concat([prev_dat, curr_dat])

#     # i = 0
#     for i in range(len(curr_aso)):
#         print(i)

#         # If first observation get all obs previously
#         if i == 0:
#             date_ = curr_aso.loc[[i], 'date'][i]

#             pindat = indat[indat['date'] <= pd.to_datetime(date_)]
#             pindat = pindat.assign(lat_lon = pindat['latitude'].astype(str) + "_" + pindat['longitude'].astype(str))
#             pindat['snow'] = np.where(pindat['tmean'] < 0, np.where(pindat['ppt'] > 0, 1, 0), 0)
#             pindat = pindat[pindat['snow'] == 1]
#             pindat = pindat.groupby('lat_lon').agg({'snow': np.sum, 'tmean': np.sum, 'tmax': np.sum, 'tmin': np.sum, 'ppt': np.sum, 'gridNumber': np.mean}).reset_index().drop(columns=['lat_lon'])
#             pindat = pindat.assign(aso_date = date_)    
#             dat_.append(pindat)

#         # If not first, get observations between i-1 and i
#         else:
#             date_ = curr_aso.loc[[i], 'date'][i]
#             pdate_ = curr_aso.loc[[i-1], 'date'][i-1]

#             pindat = indat[(indat['date'] <= pd.to_datetime(date_)) & (indat['date'] > pd.to_datetime(pdate_))]
#             pindat = pindat.assign(lat_lon = pindat['latitude'].astype(str) + "_" + pindat['longitude'].astype(str))
#             pindat['snow'] = np.where(pindat['tmean'] < 0, np.where(pindat['ppt'] > 0, 1, 0), 0)
#             pindat = pindat[pindat['snow'] == 1]
#             pindat = pindat.groupby('lat_lon').agg({'snow': np.sum, 'tmean': np.sum, 'tmax': np.sum, 'tmin': np.sum, 'ppt': np.sum, 'gridNumber': np.mean}).reset_index().drop(columns=['lat_lon'])
#             pindat = pindat.assign(aso_date = date_)    
#             dat_.append(pindat)

# aso_prism = pd.concat(dat_)


# Merg elev
mdat = aso_dat.merge(aso_elev, on=['lat_lon'], how='left')
mdat
len(mdat)

len(mdat) == len(mdat.dropna())
len(mdat) == len(mdat.drop_duplicates())


[check_unique_lat_lon_by_date(mdat, x) for x in [2013, 2016, 2017, 2018, 2019, 2020, 2021]]



# Merge Prism
aso_prism_lookup = pd.read_csv("data/Tuolumne_Watershed/aso_prism_lookup.csv")
aso_prism_lookup = aso_prism_lookup.assign(lat_lon = aso_prism_lookup['lat'].astype(str) + "_" + aso_prism_lookup['lon'].astype(str))
aso_prism_lookup = aso_prism_lookup.drop_duplicates(subset=['lat_lon'])
aso_prism_lookup = aso_prism_lookup[['lat_lon', 'prism_grid']]

mdat = mdat.merge(aso_prism_lookup, on='lat_lon', how='left')

# prism_dat = prism_dat.drop_duplicates(subset=['gridNumber', 'aso_date'])

# prism_dat['aso_date'] = pd.to_datetime(prism_dat['aso_date'])
# mdat['date'] = pd.to_datetime(mdat['date'])

mdat = mdat.merge(prism_dat, left_on=['date', 'prism_grid'], right_on=['aso_date', 'gridNumber'], how='left')

[check_unique_lat_lon_by_date(mdat, x) for x in [2013, 2016, 2017, 2018, 2019, 2020, 2021]]

mdat.columns

mdat.to_parquet("data/Tuolumne_Watershed/model_data_elevation_prism_sinceSep_V2.parquet", compression=None)


# Merge NLCD
ldat = pd.read_csv("data/Tuolumne_Watershed/aso_nlcd_lookup.csv")
ldat = ldat.assign(lat = np.round(ldat['lat'], 4),
                         lon = np.round(ldat['lon'], 4))
ldat = ldat.assign(lat_lon = ldat['lat'].astype(str) + "_" + ldat['lon'].astype(str))
ldat = ldat.drop_duplicates(subset='lat_lon')
ldat = ldat[['lat_lon', 'nlcd_grid']]

mdat = mdat.merge(ldat, on='lat_lon', how='left')

[check_unique_lat_lon_by_date(mdat, x) for x in mdat['year'].unique()]  

ldat_2013 = pd.read_csv("data/NLCD/processed/nlcd_2013_land_cover_l48_20210604.csv")
ldat_2013 = ldat_2013.assign(lat_lon = ldat_2013['lat'].astype(str) + "_" + ldat_2013['lon'].astype(str))
ldat_2013 = ldat_2013.drop(columns=['year'])

ldat_2016 = pd.read_csv("data/NLCD/processed/nlcd_2016_land_cover_l48_20210604.csv")
ldat_2016 = ldat_2016.assign(lat_lon = ldat_2016['lat'].astype(str) + "_" + ldat_2016['lon'].astype(str))
ldat_2016 = ldat_2016.drop(columns=['year'])

ldat_2019 = pd.read_csv("data/NLCD/processed/nlcd_2019_land_cover_l48_20210604.csv")
ldat_2019 = ldat_2019.assign(lat_lon = ldat_2019['lat'].astype(str) + "_" + ldat_2019['lon'].astype(str))
ldat_2019 = ldat_2019.drop(columns=['year'])

# mdat['year'] = pd.to_datetime(mdat['date']).dt.year

# Model data 2013
mdat1 = mdat[mdat['year'] <= 2013]

[check_unique_lat_lon_by_date(mdat1, x) for x in mdat1['year'].unique()]

# Merge NLCD
mdat1 = mdat1.drop(columns=['lat', 'lon'])
mdat1 = mdat1.merge(ldat_2013, left_on='nlcd_grid', right_on='lat_lon', how='left')

mdat1 = mdat1.rename(columns={'lat_lon_x': 'lat_lon'})
mdat1 = mdat1.drop(columns=['lat_lon_y'])

[check_unique_lat_lon_by_date(mdat1, x) for x in mdat1['year'].unique()]

# Model data 2016, 2017
mdat2 = mdat[(mdat['year'] == 2016) | (mdat['year'] == 2017)]

# Merge NLCD
mdat2 = mdat2.drop(columns=['lat', 'lon'])
mdat2 = mdat2.merge(ldat_2016, left_on='nlcd_grid', right_on='lat_lon', how='left')

mdat2 = mdat2.rename(columns={'lat_lon_x': 'lat_lon'})
mdat2 = mdat2.drop(columns=['lat_lon_y'])

[check_unique_lat_lon_by_date(mdat2, x) for x in mdat2['year'].unique()]

# Model data 2018-2022
mdat3 = mdat[mdat['year'] >= 2018]

# Merge NLCD
mdat3 = mdat3.drop(columns=['lat', 'lon'])
mdat3 = mdat3.merge(ldat_2019, left_on='nlcd_grid', right_on='lat_lon', how='left')

mdat3 = mdat3.rename(columns={'lat_lon_x': 'lat_lon'})
mdat3 = mdat3.drop(columns=['lat_lon_y'])

[check_unique_lat_lon_by_date(mdat3, x) for x in mdat3['year'].unique()]

mdat_nlcd = pd.concat([mdat1, mdat2, mdat3], axis=0)

mdat_nlcd['lat_x_lon'] = mdat_nlcd['lat'] * mdat_nlcd['lon']

[check_unique_lat_lon_by_date(mdat_nlcd, x) for x in mdat_nlcd['year'].unique()]

save_dat = mdat_nlcd[mdat_nlcd['year'] <= 2021]

save_dat.columns


# mdat_nlcd = mdat_nlcd.iloc[:, [0, 1, 2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]


save_dat = save_dat[['date', 'lat_lon_x', 'SWE', 'lat', 'lon', 
       'prism_grid', 'snow', 'tmean', 'tmax', 'tmin', 'ppt', 'gridNumber',
       'aso_date', 'elevation', 'year_x', 'month',
       'lat_x_lon', 'nlcd_grid', 'landcover']]

save_dat.columns = ['date', 'lat_lon', 'SWE', 'lat', 'lon', 
       'prism_grid', 'snow', 'tmean', 'tmax', 'tmin', 'ppt', 'gridNumber',
       'aso_date', 'elevation', 'year', 'month',
       'lat_x_lon', 'nlcd_grid', 'landcover']


len(save_dat) == len(save_dat.dropna())
len(save_dat) == len(save_dat.drop_duplicates())

[check_unique_lat_lon_by_date(save_dat, x) for x in [2013, 2016, 2017, 2018, 2019, 2020, 2021]]

save_dat.to_parquet("data/Tuolumne_Watershed/model_data_elevation_prism_sinceSep_nlcd_V2.parquet", compression=None)


# CHECKS!

def check_unique_lat_lon_by_date(dataframe, year):
    dataframe['date'] = pd.to_datetime(dataframe['date'])
    
    # Filter the DataFrame for the specified year
    year_data = dataframe[dataframe['date'].dt.year == year]
    
    # Check if all "lat_lon" combinations are unique within the specified year
    unique_date_count_per_group = year_data.groupby('date')['lat_lon'].nunique()

    is_unique = unique_date_count_per_group.sum() == len(year_data)

    return is_unique

