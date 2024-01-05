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
# import gdal
# from osgeo import gdal
# from osgeo import gdal_array
# from osgeo import gdalconst
# from bs4 import BeautifulSoup

import dask.dataframe as dd
from dask import delayed
from dask.diagnostics import ProgressBar

from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

# import ee

from haversine import haversine

from libs.asolibs import *
from libs.prismlibs import *

tqdm.pandas()


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


def get_elevation(lat, lon, num_retries=10):
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
            time.sleep(10)  # wait for 1 second before retrying
    print(f"Failed to retrieve elevation after {num_retries} attempts")
    return "NA"


def elevation_function(df, lat_column, lon_column, main_directory, max_workers=8):
    """Query service using lat, lon. Save individual files for each coordinate."""

    def save_elevation(lat, lon):
        file_name =  f"{lat:.4f}_{lon:.4f}.csv"
        file_path = os.path.join(main_directory, file_name)

        # Check if the file already exists
        if os.path.isfile(file_path):
            # Uncomment the next line if you want to print a message for existing files
            # print(f"File '{file_name}' already exists. Skipping.")
            return

        elevation = get_elevation(lat, lon)

        if elevation == "NA":
            return
        
        # Save the elevation as a separate file
        with open(file_path, 'w') as file:
            file.write("Latitude,Longitude,Elevation\n")
            file.write(f"{lat},{lon},{elevation}\n")

    def fetch_elevation(row):
        lat = row[1][lat_column]
        lon = row[1][lon_column]
    
        save_elevation(lat, lon)

    # Using ThreadPoolExecutor to parallelize the requests
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Wrap the executor.map with tqdm for a progress bar
        _ = list(tqdm(executor.map(fetch_elevation, df.iterrows()), total=len(df), desc="Fetching elevation data"))


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
        rdat.to_csv(f"data/{basin_name}/NDVI/{year}.csv", index=False)
    else:
        print(f"No data available for year {year}")


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


def find_closest_grid_prism_parallel(row):
    return find_closest_grid(row['lat'], row['lon'], pdat, 'gridNumber')


def find_closest_grid_nlcd_parallel(row):
    return find_closest_grid(row['lat'], row['lon'], ndat, 'lat_lon', decimal=0.001)


def find_closest_grid_elev_grade_aspect_parallel(row):
    return find_closest_grid(row['lat'], row['lon'], edat, 'index', decimal=0.01)

lat = 37.7396
lon = -119.2703
dat = ndat
return_column = "lat_lon"
decimal=0.001

def find_closest_grid(lat, lon, dat, return_column, decimal=0.1):
    min_distance = np.inf
    closest_value = None

    dat = dat[(dat['lat'] >= lat - decimal) & (dat['lat'] <= lat + decimal)]
    dat = dat[(dat['lon'] >= lon - decimal) & (dat['lon'] <= lon + decimal)]

    for idx in dat.index:
        lat_p, lon_p = dat.at[idx, 'lat'], dat.at[idx, 'lon']
        distance = haversine((lat, lon), (lat_p, lon_p))

        if distance < min_distance:
            min_distance = distance
            closest_value = dat.at[idx, return_column]

    return closest_value


def get_nlcd(filename, shape_loc):

    print(f"Processing: {filename}")

    # Get year
    year = filename.split("_")[1]

    save_filename = filename.split("/")[-1].split(".")[0]

    # Open the dataset
    ds = rxr.open_rasterio(f"data/NLCD/raw/{filename}")

    # Reproject to EPSG:4326
    ds = ds.rio.reproject("EPSG:4326")

    # Load the shapefile
    # shape_loc = f"data/shapefiles/{basin_name}/Tuolumne_Watershed.shp"

    # target_crs_wkt = 'PROJCS["Albers Conical Equal Area",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Albers_Conic_Equal_Area"],PARAMETER["latitude_of_center",23],PARAMETER["longitude_of_center",-96],PARAMETER["standard_parallel_1",29.5],PARAMETER["standard_parallel_2",45.5],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["meters",1],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
    gdf = gpd.read_file(shape_loc)
    # gdf = gdf.to_crs(target_crs_wkt)
    # gdf = gdf.to_crs(epsg="4326")
    minx = gdf.bounds['minx'][0] - 1
    maxx = gdf.bounds['maxx'][0] + 1
    miny = gdf.bounds['miny'][0] - 1
    maxy = gdf.bounds['maxy'][0] + 1

    ds = ds.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)

    outdat = ds.rio.clip(gdf.geometry, all_touched=True, drop=True, invert=False, from_disk=True) 

    outdat = outdat.to_series().reset_index()
    outdat.columns = ['band', 'lat', 'lon', 'landcover']
    outdat = outdat.assign(year = year)
    outdat = outdat[['year', 'lat', 'lon', 'landcover']]

    # Filter no class
    outdat = outdat[outdat['landcover'] != 0]
    outdat = outdat[outdat['landcover'] != 255]

    outdat.to_csv(f"data/{basin_name}/processed/{save_filename}.csv", index=False)
    print(f"Saved: data/{basin_name}/processed/{save_filename}.csv")


def proc_elevation(gdf, basin_name):
    
    print(f"Getting ASO data for {basin_name}")
    
    # Get unique lat/lon
    lat_lon_dat = pd.read_csv(f"data/{basin_name}/processed/aso_basin_data.csv")
    lat_lon_dat = lat_lon_dat.drop_duplicates(subset=['lat_lon'])

    # lat_lon_dat = lat_lon_dat.iloc[0:20, :]

    # print("Checking for already processed files")
    elev_files = glob.glob(f"data/{basin_name}/elevation/*.csv")
    lat_lon_files = [x.split("/")[-1].replace(".csv", "") for x in elev_files] 

    lat_lon_dat = lat_lon_dat[~lat_lon_dat['lat_lon'].isin(lat_lon_files)]

    if len(lat_lon_dat) > 0:
        print(f"Processing ASO elevation data")
        elevation_function(lat_lon_dat, 'lat', 'lon', f"data/{basin_name}/elevation/", 50)
    else:
        print("No elevations to process. Binding data")

    print(f"Binding elevation data")
    elev_files = glob.glob(f"data/{basin_name}/elevation/*.csv")

    len(elev_files)
    len(lat_lon_dat['lat_lon'].unique())    

    df_ = []
    for file_ in tqdm(elev_files):
        df = pd.read_csv(file_)
        df_.append(df)

    eldat = pd.concat(df_)
    eldat.to_parquet(f"data/{basin_name}/processed/aso_elevation.parquet", compression=None, index=False)
    print(f"Saved: data/{basin_name}/processed/aso_elevation.parquet")


def proc_basin_prism(gdf, basin_name, min_year, max_year):

    # Generate prism data between years
    proc_daily(gdf, basin_name, min_year, max_year)

    # pdat = pd.read_csv(f"data/{basin_name}/processed/{basin_name}_PRISM_daily_{min_year}-{max_year}.csv")
    # pdat = pdat.drop_duplicates(subset=['gridNumber'])
    # pdat = pdat[['longitude', 'latitude', 'gridNumber']]
    # pdat.columns = ['lon', 'lat', 'gridNumber']

    # gdf = gpd.read_file(shape_loc)
    # gdf = gdf.to_crs(epsg=4326)

    # ldat = pd.read_csv(f"data/{basin_name}/processed/aso_basin_data.csv")
    # ldat = ldat[['lon', 'lat']].reset_index(drop=True)
    # ldat = ldat.assign(lat_lon = ldat['lat'].astype(str) + "_" + ldat['lon'].astype(str))


def proc_elev_grade_aspect_lookup(basin_name, shape_loc):

    global edat

    edat = pd.read_csv(f"data/{basin_name}/elev_grade_aspect/elev_grade_aspect.csv")
    # edat = edat.assign(lat_lon = edat['lat'].astype(str) + "_" + edat['lon'].astype(str)) 
    edat['lat_lon'] = edat['lat'].apply(lambda x: f"{x:.{4}f}") + "_" + edat['lon'].apply(lambda x: f"{x:.{4}f}")

    edat = edat.reset_index()

    gdf = gpd.read_file(shape_loc)
    gdf = gdf.to_crs(epsg=4326)

    ldat = pd.read_csv(f"data/{basin_name}/processed/aso_basin_data.csv")
    ldat = ldat[['lon', 'lat']].reset_index(drop=True)

    ldat = ldat.assign(lat = np.round(ldat['lat'], 4),
                             lon = np.round(ldat['lon'], 4))

    ldat['lat_lon'] = ldat['lat'].apply(lambda x: f"{x:.{4}f}") + "_" + ldat['lon'].apply(lambda x: f"{x:.{4}f}")
    # ldat = ldat.assign(lat_lon = ldat['lat'].astype(str) + "_" + ldat['lon'].astype(str))
    ldat = ldat.drop_duplicates(subset='lat_lon')

    # ldat = ldat.iloc[0:20, :]

    # Use apply to find the closest grid for all rows
    # Convert DataFrame to list of dictionaries
    rows = ldat.to_dict(orient='records')

    # Set the number of workers
    num_workers = multiprocessing.cpu_count()

    # Initialize multiprocessing Pool
    with multiprocessing.Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(find_closest_grid_elev_grade_aspect_parallel, rows), total=len(rows), desc="Processing:"))

    ldat['elev_grade_aspect_grid'] = results
    
    merge_dat = edat[['index', 'elev_m', 'slope', 'aspect']]

    ldat = ldat.merge(merge_dat, how='left', left_on='elev_grade_aspect_grid', right_on='index')

    ldat = ldat[['lon', 'lat', 'lat_lon', 'elev_m', 'slope', 'aspect']]

    ldat.to_csv(f"data/{basin_name}/processed/aso_elev_grade_aspect.csv", index=False)


def proc_prism_lookup(basin_name, shape_loc):

    prism_loc = glob.glob(f"data/{basin_name}/processed/*PRISM*")
    pdat = pd.read_csv(prism_loc[0])
    pdat = pdat.drop_duplicates(subset=['gridNumber'])
    pdat = pdat[['longitude', 'latitude', 'gridNumber']]
    pdat.columns = ['lon', 'lat', 'gridNumber']

    gdf = gpd.read_file(shape_loc)
    gdf = gdf.to_crs(epsg=4326)

    ldat = pd.read_csv(f"data/{basin_name}/processed/aso_basin_data.csv")
    ldat = ldat[['lon', 'lat']].reset_index(drop=True)

    ldat = ldat.assign(lat = np.round(ldat['lat'], 4),
                             lon = np.round(ldat['lon'], 4))

    ldat = ldat.assign(lat_lon = ldat['lat'].astype(str) + "_" + ldat['lon'].astype(str))
    ldat = ldat.drop_duplicates(subset='lat_lon')

    # Use apply to find the closest grid for all rows
    ldat['prism_grid'] = ldat.progress_apply(lambda row: find_closest_grid(row['lat'], row['lon'], pdat, 'gridNumber'), axis=1)
    ldat.to_csv(f"data/{basin_name}/processed/aso_prism_lookup.csv", index=False)


def proc_nlcd(gdf, shape_loc):
    # ----------------------------------
    # Land cover
    nlcd_files  = ["nlcd_2019_land_cover_l48_20210604.img",
                   "nlcd_2016_land_cover_l48_20210604.img",
                   "nlcd_2013_land_cover_l48_20210604.img"]


    [get_nlcd(x, shape_loc) for x in nlcd_files]

    ldat = pd.read_csv(f"data/{basin_name}/processed/aso_basin_data.csv")
    ldat = ldat.drop_duplicates(subset='lat_lon')
    ldat = ldat[['lon', 'lat']].reset_index(drop=True)
    
    # ldat.to_csv(f"data/{basin_name}/unique_ASO_grids.csv", index=False)

    # ldat = pd.read_csv(f"data/{basin_name}/unique_ASO_grids.csv")

    global ndat

    ndat = pd.read_csv(f"data/{basin_name}/processed/nlcd_2019_land_cover_l48_20210604.csv")
    ndat = ndat.assign(lat_lon = ndat['lat'].astype(str) + "_" + ndat['lon'].astype(str))

    # Convert DataFrame to list of dictionaries
    rows = ldat.to_dict(orient='records')

    # Set the number of workers
    num_workers = multiprocessing.cpu_count()

    # Initialize multiprocessing Pool
    with multiprocessing.Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(find_closest_grid_nlcd_parallel, rows), total=len(rows), desc="Processing:"))

    # Assign the results to the DataFrame
    ldat['nlcd_grid'] = results
    len(ldat.dropna()) == len(ldat)
    ldat.to_csv(f"data/{basin_name}/processed/aso_nlcd_lookup.csv", index=False)


def bind_data(basin_name, shape_loc, min_year, max_year):

    decimal_point = 4
    # Load the shapefile using geopandas and convert its coordinate reference system to WGS84
    gdf = gpd.read_file(shape_loc)
    gdf = gdf.to_crs(epsg=4326)

    # Process ASO SWE data, drop missing values, and filter out negative SWE values
    aso_dat = pd.read_csv(f"data/{basin_name}/processed/aso_basin_data.csv")
    
    # Extract year and month from date, convert to integer, and assign to new columns
    aso_dat['year'] = pd.to_datetime(aso_dat['date']).dt.year
    aso_dat['month'] = pd.to_datetime(aso_dat['date']).dt.month

    # Check unique latitude and longitude by date
    [check_unique_lat_lon_by_date(aso_dat, x) for x in aso_dat['year'].unique()]

    # Load elevation data from a parquet file, rename columns, and create lat_lon column
    # OLD elevation
    # aso_elev = pd.read_parquet(f"data/{basin_name}/processed/aso_elevation.parquet")
    # aso_elev.columns = ['lat', 'lon', 'elevation']

    aso_elev = pd.read_csv(f"data/{basin_name}/processed/aso_elev_grade_aspect.csv")
    aso_elev.columns = ['lat', 'lon', 'lat_lon', 'elevation', 'slope', 'aspect']

    # aso_elev['lat_lon'] = aso_elev['lat'].apply(lambda x: f"{x:.{decimal_point}f}") + "_" + aso_elev['lon'].apply(lambda x: f"{x:.{decimal_point}f}")

    # Drop original latitude and longitude columns as they are no longer needed
    aso_elev = aso_elev.drop(columns=['lat', 'lon'])

    aso_elev = aso_elev.drop_duplicates()

    # Ensure all lat_lon values in elevation data are unique
    assert len(aso_elev['lat_lon'].unique()) == len(aso_elev), "Lat_Lon values are not unique!"

    # --------------------------------------------------------------------------
    # Merge Prism

    # Load PRISM data, convert date to datetime, and extract month and year
    prism_dat_loc = glob.glob(f"data/{basin_name}/processed/*PRISM_daily*")[0]
    prism_dat = pd.read_csv(prism_dat_loc)
    prism_dat['date'] = pd.to_datetime(prism_dat['date'], format="%Y%m%d")
    prism_dat['month'] = prism_dat['date'].dt.month
    prism_dat['year'] = prism_dat['date'].dt.year

    # Pivot the PRISM data to wide format and calculate the mean temperature
    prism_dat = prism_dat.pivot_table(index=['date', 'gridNumber', 'month', 'year'], columns='var', values='value', aggfunc=np.nanmean).reset_index()
    prism_dat['tmean'] = (prism_dat['tmax'] + prism_dat['tmin']) / 2

    # Get data from the previous year (ensure this function is defined in your code)
    prism_dat = get_data_from_prev_year(aso_dat, prism_dat)

    # Check for missing values in PRISM data
    assert len(prism_dat) == len(prism_dat.dropna()), "Missing values in PRISM data"
    assert prism_dat.groupby('aso_date').count().sum()['gridNumber'] == len(prism_dat['gridNumber'].unique()) * len(prism_dat['aso_date'].unique()), "Data grouping validation failed"

    # Main data merge with elevation data
    mdat = aso_dat.merge(aso_elev, on='lat_lon', how='left')

    assert len(mdat) == len(mdat.dropna()), "Missing values after merging with elevation data"
    assert len(mdat) == len(mdat.drop_duplicates()), "Duplicate values after merging with elevation data"

    # Check unique latitude and longitude by date
    [check_unique_lat_lon_by_date(mdat, x) for x in mdat['year'].unique()]

    # Load ASO PRISM lookup data, create lat_lon identifier, and merge with main data
    aso_prism_lookup = pd.read_csv(f"data/{basin_name}/processed/aso_prism_lookup.csv")
    # aso_prism_lookup['lat_lon'] = aso_prism_lookup['lat'].astype(str) + "_" + aso_prism_lookup['lon'].astype(str)

    aso_prism_lookup['lat_lon'] = aso_prism_lookup['lat'].apply(lambda x: f"{x:.{decimal_point}f}") + "_" + aso_prism_lookup['lon'].apply(lambda x: f"{x:.{decimal_point}f}")

    aso_prism_lookup = aso_prism_lookup.drop_duplicates(subset='lat_lon')[['lat_lon', 'prism_grid']]
    mdat = mdat.merge(aso_prism_lookup, on='lat_lon', how='left')

    # Merge PRISM data with main data
    mdat = mdat.merge(prism_dat, left_on=['date', 'prism_grid'], right_on=['aso_date', 'gridNumber'], how='left')

    mdat = mdat[mdat['year'] <= max_year]

    # Check unique latitude and longitude by date after merging PRISM data
    [check_unique_lat_lon_by_date(mdat, x) for x in mdat['year'].unique()]

    assert len(mdat) == len(mdat.dropna()), "Missing values after merging with prism data"
    assert len(mdat) == len(mdat.drop_duplicates()), "Duplicate values after merging with prism data"

    # Save the merged data to a parquet file
    mdat.to_parquet(f"data/{basin_name}/processed/model_data_elevation_prism_sinceSep.parquet", compression=None)

    # mdat = pd.read_parquet(f"data/{basin_name}/processed/model_data_elevation_prism_sinceSep.parquet")

    # --------------------------------------------------------------------------
    # NLCD Data Merge

    # Load NLCD lookup data, round latitude and longitude to 4 decimal places,
    # and create a new identifier combining latitude and longitude
    ldat = pd.read_csv(f"data/{basin_name}/processed/aso_nlcd_lookup.csv")
    # ldat = ldat.assign(lat=np.round(ldat['lat'], 4), lon=np.round(ldat['lon'], 4))

    ldat['lat_lon'] = ldat['lat'].apply(lambda x: f"{x:.{decimal_point}f}") + "_" + ldat['lon'].apply(lambda x: f"{x:.{decimal_point}f}")

    # ldat['lat_lon'] = ldat['lat'].astype(str) + "_" + ldat['lon'].astype(str)
    ldat = ldat.drop_duplicates(subset='lat_lon')[['lat_lon', 'nlcd_grid']]

    # Merge the NLCD lookup data with main data on the 'lat_lon' field using a left join
    mdat = mdat.merge(ldat, on='lat_lon', how='left')

    # Check for unique latitude and longitude by date after merging NLCD data
    [check_unique_lat_lon_by_date(mdat, x) for x in mdat['year'].unique()]

    # Load NLCD land cover data for the years 2013, 2016, and 2019
    # and create a 'lat_lon' identifier for each dataset
    ldat_2013 = pd.read_csv(f"data/{basin_name}/processed/nlcd_2013_land_cover_l48_20210604.csv")
    ldat_2013['lat_lon'] = ldat_2013['lat'].astype(str) + "_" + ldat_2013['lon'].astype(str)
    ldat_2013 = ldat_2013.drop(columns=['year'])

    ldat_2016 = pd.read_csv(f"data/{basin_name}/processed/nlcd_2016_land_cover_l48_20210604.csv")
    ldat_2016['lat_lon'] = ldat_2016['lat'].astype(str) + "_" + ldat_2016['lon'].astype(str)
    ldat_2016 = ldat_2016.drop(columns=['year'])

    ldat_2019 = pd.read_csv(f"data/{basin_name}/processed/nlcd_2019_land_cover_l48_20210604.csv")
    ldat_2019['lat_lon'] = ldat_2019['lat'].astype(str) + "_" + ldat_2019['lon'].astype(str)
    # ldat_2019['lat_lon'] = ldat_2019['lat'].apply(lambda x: f"{x:.{decimal_point}f}") + "_" + ldat_2019['lon'].apply(lambda x: f"{x:.{decimal_point}f}")

    ldat_2019 = ldat_2019.drop(columns=['year'])

    # Segment the main data into three parts based on the year and merge with respective NLCD data
    # Model data for years up to and including 2013
    mdat1 = mdat[mdat['year'] <= 2013]

    [check_unique_lat_lon_by_date(mdat1, x) for x in mdat1['year'].unique()]  # Check for unique lat-lon by date

    mdat1 = mdat1.drop(columns=['lat', 'lon']).merge(ldat_2013, left_on='nlcd_grid', right_on='lat_lon', how='left')
    mdat1 = mdat1.rename(columns={'lat_lon_x': 'lat_lon'}).drop(columns=['lat_lon_y'])

    # Model data for years 2016 and 2017
    mdat2 = mdat[(mdat['year'] == 2016) | (mdat['year'] == 2017)]

    [check_unique_lat_lon_by_date(mdat2, x) for x in mdat2['year'].unique()]  # Check for unique lat-lon by date

    mdat2 = mdat2.drop(columns=['lat', 'lon']).merge(ldat_2016, left_on='nlcd_grid', right_on='lat_lon', how='left')
    mdat2 = mdat2.rename(columns={'lat_lon_x': 'lat_lon'}).drop(columns=['lat_lon_y'])

    # Model data for years 2018 to 2022
    mdat3 = mdat[mdat['year'] >= 2018]

    [check_unique_lat_lon_by_date(mdat3, x) for x in mdat3['year'].unique()]  # Check for unique lat-lon by date

    mdat3 = mdat3.drop(columns=['lat', 'lon']).merge(ldat_2019, left_on='nlcd_grid', right_on='lat_lon', how='left')
    mdat3 = mdat3.rename(columns={'lat_lon_x': 'lat_lon'}).drop(columns=['lat_lon_y'])

    # Concatenate the three segments of data back together
    mdat_nlcd = pd.concat([mdat1, mdat2, mdat3], axis=0)

    # Create a new feature by multiplying latitude and longitude
    mdat_nlcd['lat_x_lon'] = mdat_nlcd['lat'] * mdat_nlcd['lon']

    # Check for unique latitude and longitude by date on the concatenated data
    [check_unique_lat_lon_by_date(mdat_nlcd, x) for x in mdat_nlcd['year'].unique()]

    # Filter the data for years up to and including 2021
    save_dat = mdat_nlcd[mdat_nlcd['year'] <= max_year]

    save_dat = save_dat[['date', 'lat_lon', 'SWE', 'lat', 'lon', 
           'prism_grid', 'snow', 'tmean', 'tmax', 'tmin', 'ppt', 'gridNumber',
           'aso_date', 'elevation', 'slope', 'aspect', 'year', 'month',
           'lat_x_lon', 'nlcd_grid', 'landcover']]

    save_dat.columns = ['date', 'lat_lon', 'SWE', 'lat', 'lon', 
           'prism_grid', 'snow', 'tmean', 'tmax', 'tmin', 'ppt', 'gridNumber',
           'aso_date', 'elevation', 'slope', 'aspect', 'year', 'month',
           'lat_x_lon', 'nlcd_grid', 'landcover']

    assert len(save_dat) == len(save_dat.dropna())

    save_dat.to_parquet(f"data/{basin_name}/processed/model_data_elevation_prism_sinceSep_nlcd.parquet", compression=None)
    print(f"Saved: data/{basin_name}/processed/model_data_elevation_prism_sinceSep_nlcd.parquet")


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


def proc_aso_swe(gdf, basin_name, decimal_point=4):

    aso = proc_ASO_SWE_shp(gdf)
    aso = aso.dropna()
    aso = aso[aso['SWE'] >= 0]

    print("Defining grids")
    lat_lon_dat = aso.assign(lat = np.round(aso['lat'], decimal_point),
        lon = np.round(aso['lon'], decimal_point))

    # lat_lon_dat['lat_lon'] = lat_lon_dat['lat'].astype(str) + "_" + lat_lon_dat['lon'].astype(str)
    
    lat_lon_dat['lat_lon'] = lat_lon_dat['lat'].apply(lambda x: f"{x:.{decimal_point}f}") + "_" + lat_lon_dat['lon'].apply(lambda x: f"{x:.{decimal_point}f}")

    print("Grouping by and getting averages")
    lat_lon_dat = lat_lon_dat.groupby(['date', 'site', 'lat_lon']).agg({'SWE': np.mean, 'lat': np.mean, 'lon': np.mean}).reset_index()

    lat_lon_dat.to_csv(f"data/{basin_name}/processed/aso_basin_data.csv", index=False)




basin_name = "Tuolumne_Watershed"
min_year = 1981
max_year = 2021

basin_name = "Blue_Dillon_Watershed"
min_year = 1981
max_year = 2023

basin_name = "Dolores_Watershed"
min_year = 1981
max_year = 2022


basin_name = "Conejos_Watershed"
min_year = 1981
max_year = 2022


def main(basin_name, min_year, max_year):
    
    shape_loc = glob.glob(f"data/{basin_name}/shapefiles/*.shp")[0]
    
    gdf = gpd.read_file(shape_loc)
    gdf = gdf.to_crs(epsg=4326)

    setup_basin(basin_name)

    proc_aso_swe(gdf, basin_name)

    # TO DO --- ADD FUNCTION TO PROCESS THIS
    proc_grade_elev_watershed(gdf, basin_name)

    proc_elev_grade_aspect_lookup(basin_name, shape_loc)

    # proc_elevation(gdf, basin_name)

    proc_basin_prism(gdf, basin_name, min_year, max_year)

    proc_prism_lookup(basin_name, shape_loc)

    proc_nlcd(gdf, shape_loc)

    bind_data(basin_name, shape_loc, min_year=min_year, max_year=max_year)




if __name__ == "__main__":

    fire.Fire(main)

    # basin_name = "Blue_Dillon_Watershed"
    # shape_loc = glob.glob(f"data/{basin_name}/shapefiles/*.shp")[0]
    
    # min_year = 1981
    # max_year = 2021

    # gdf = gpd.read_file(shape_loc)
    # gdf = gdf.to_crs(epsg=4326)

    # setup_basin(basin_name)

    # proc_aso_swe(gdf, basin_name, decimal_point)

    # proc_elevation(gdf, basin_name)

    # fp = gpd.read_file('data/Tuolumne_Watershed/shapefiles/Tuolumne_Watershed.shp')
    # proc_grade_elev_watershed("Tuolumne_Watershed", fp)

    # fp = gpd.read_file('data/Blue_Dillon_Watershed/shapefiles/Blue_Dillon.shp')
    # proc_grade_elev_watershed("Blue_Dillon_Watershed", fp)
    
    # fp = gpd.read_file('data/Conejos_Watershed/shapefiles/Conejos_waterbasin.shp')
    # proc_grade_elev_watershed("Conejos_Watershed", fp)
    
    # fp = gpd.read_file('data/Dolores_Watershed/shapefiles/Dolores_waterbasin.shp')
    # proc_grade_elev_watershed("Dolores_Watershed", fp)


    # proc_basin_prism(gdf, basin_name, min_year, max_year)

    # proc_prism_lookup(basin_name, shape_loc)

    # proc_nlcd(gdf, shape_loc)

    # bind_data(basin_name, shape_loc, min_year=1981, max_year=2023)
