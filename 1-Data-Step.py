import os
import urllib
import urllib.request
import time
import multiprocessing
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from shapely.geometry import Point

import requests
import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
import xarray as xr
import rioxarray as rxr

from osgeo import gdal
from osgeo import gdal_array
from osgeo import gdalconst

import dask.dataframe as dd
from dask import delayed
from dask.diagnostics import ProgressBar
from tqdm import tqdm
from haversine import haversine

from libs.asolibs import *
from libs.prismlibs import *
from libs.elevgradeaspectlibs import *
from libs.nlcdlibs import *
from misclibs import *

tqdm.pandas()


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







basin_name = "Tuolumne_Watershed"
min_year = 1981
max_year = 2023

basin_name = "Blue_Dillon_Watershed"
min_year = 1981
max_year = 2023

basin_name = "Dolores_Watershed"
min_year = 1981
max_year = 2022


basin_name = "Conejos_Watershed"
min_year = 1981
max_year = 2022

# # Tuolumne Basin shapefile
# fp = gpd.read_file('data/Tuolumne_Watershed/shapefiles/Tuolumne_Watershed.shp')
# proc_grade_elev_watershed("Tuolumne_Watershed", fp)

# fp = gpd.read_file('data/Blue_Dillon_Watershed/shapefiles/Blue_Dillon.shp')
# proc_grade_elev_watershed("Blue_Dillon_Watershed", fp)

# fp = gpd.read_file('data/Conejos_Watershed/shapefiles/Conejos_waterbasin.shp')
# proc_grade_elev_watershed("Conejos_Watershed", fp)

# fp = gpd.read_file('data/Dolores_Watershed/shapefiles/Dolores_waterbasin.shp')
# proc_grade_elev_watershed("Dolores_Watershed", fp)






def main(basin_name, min_year, max_year):
    
    shape_loc = glob.glob(f"data/{basin_name}/shapefiles/*.shp")[0]
    
    gdf = gpd.read_file(shape_loc)
    gdf = gdf.to_crs(epsg=4326)

    setup_basin(basin_name)

    proc_aso_swe(gdf, basin_name)

    # TO DO --- ADD FUNCTION TO PROCESS THIS
    proc_grade_elev_watershed(basin_name, gdf)

    proc_elev_grade_aspect_lookup(basin_name, shape_loc)

    #### NOT IN USE
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
