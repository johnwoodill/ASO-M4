import pandas as pd 
import numpy as np
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
from keras.models import load_model
import joblib

from libs.asolibs import *


def check_unique_lat_lon_by_date(dataframe, year):
    dataframe['date'] = pd.to_datetime(dataframe['date'])
    
    # Filter the DataFrame for the specified year
    year_data = dataframe[dataframe['date'].dt.year == year]
    
    # Check if all "lat_lon" combinations are unique within the specified year
    unique_date_count_per_group = year_data.groupby('date')['lat_lon'].nunique()

    is_unique = unique_date_count_per_group.sum() == len(year_data)

    return is_unique



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



def generate_dates(start_year, end_year, month, day):

    dates = [f"{year}-{month:02d}-{day:02d}" for year in range(start_year, end_year+1)]
    
    df = pd.DataFrame({'date': dates})
    df['date'] = pd.to_datetime(df['date'])   # Convert to datetime type
    df['year'] = df['date'].dt.year
    return df


# pred_dates = pd.DataFrame({'date': ["2010-04-01", "2011-04-01", "2012-04-01", "2013-04-01", 
#                                     "2014-04-01", "2015-04-01", "2016-04-01", "2017-04-01", 
#                                     "2018-04-01", "2019-04-01", "2020-04-01", "2021-04-01"]})    

# pred_dates['year'] = pd.to_datetime(pred_dates['date']).dt.year


pred_dates = generate_dates(2010, 2021, 4, 1)

shape_loc = "data/shapefiles/tuolumne_watershed/Tuolumne_Watershed.shp"

waterbasin = "Tuolumne_Watershed"



# Start of proc function

gdf = gpd.read_file(shape_loc)
gdf = gdf.to_crs(epsg=4326)

# Get ASO Data
aso_dat = proc_ASO_SWE_shp(gdf)
aso_dat = aso_dat.dropna()
aso_dat = aso_dat[aso_dat['SWE'] >= 0]

aso_dat = aso_dat.assign(lat = np.round(aso_dat['lat'], 4),
                         lon = np.round(aso_dat['lon'], 4))

# Different from data step
aso_dat['lat_lon'] = aso_dat['lat'].astype(str) + "_" + aso_dat['lon'].astype(str)
aso_dat = aso_dat.drop_duplicates(subset=['lat_lon'])
aso_dat = aso_dat[['site', 'lat_lon', 'lat', 'lon']]

len(aso_dat)    # 3263648
len(aso_dat.dropna())/len(aso_dat)

# Differernt from data step
aso_dat['key'] = 1
pred_dates['key'] = 1

aso_dat = aso_dat.merge(pred_dates, on='key').drop('key', axis=1)
aso_dat['year'].unique()


# Get elevation data
aso_elev = pd.read_parquet(f"data/{waterbasin}/aso_elevation.parquet")
aso_elev.columns = ['lat', 'lon', 'elevation']
aso_elev = aso_elev.assign(lat_lon = aso_elev['lat'].astype(str) + "_" + aso_elev['lon'].astype(str))
aso_elev = aso_elev[['lat_lon', 'elevation']]

# Prism data
prism_dat = pd.read_csv(f"data/{waterbasin}/{waterbasin}_PRISM_daily_1981-2020.csv")
prism_dat = prism_dat.assign(date = pd.to_datetime(prism_dat['date'], format = "%Y%m%d"))
prism_dat = prism_dat.assign(month = pd.to_datetime(prism_dat['date']).dt.strftime("%m"))
prism_dat = prism_dat.assign(year = pd.to_datetime(prism_dat['date']).dt.strftime("%Y"))

prism_dat = prism_dat.assign(month = prism_dat['month'].astype(int),
                             year = prism_dat['year'].astype(int))

prism_dat = prism_dat.pivot_table(index=['date', 'gridNumber', 'month', 'year'],
                            columns='var', aggfunc=np.nanmean,
                            values='value').reset_index()

prism_dat = prism_dat.assign(tmean = (prism_dat['tmax'] + prism_dat['tmin'])/2)


prism_dat = get_data_from_prev_year(pred_dates, prism_dat)

assert len(prism_dat) == len(prism_dat.dropna())
assert len(prism_dat) == len(prism_dat.drop_duplicates())


# Merg elev
mdat = aso_dat.merge(aso_elev, on=['lat_lon'], how='left')

assert len(mdat) == len(mdat.dropna())
assert len(mdat) == len(mdat.drop_duplicates())

print("Checking data merged correctly with elevation")
elev_check = [check_unique_lat_lon_by_date(mdat, x) for x in mdat['year'].unique()]
assert np.sum(elev_check) == len(mdat['year'].unique())


# Merge Prism
aso_prism_lookup = pd.read_csv(f"data/{waterbasin}/aso_prism_lookup.csv")
aso_prism_lookup = aso_prism_lookup.assign(lat_lon = aso_prism_lookup['lat'].astype(str) + "_" + aso_prism_lookup['lon'].astype(str))
aso_prism_lookup = aso_prism_lookup.drop_duplicates(subset=['lat_lon'])
aso_prism_lookup = aso_prism_lookup[['lat_lon', 'prism_grid']]

mdat = mdat.merge(aso_prism_lookup, on='lat_lon', how='left')

# prism_dat = prism_dat.drop_duplicates(subset=['gridNumber', 'aso_date'])

# prism_dat['aso_date'] = pd.to_datetime(prism_dat['aso_date'])
# mdat['date'] = pd.to_datetime(mdat['date'])

mdat = mdat.merge(prism_dat, left_on=['date', 'prism_grid'], right_on=['aso_date', 'gridNumber'], how='left')

print("Checking data merged correctly with PRISM")
prism_check = [check_unique_lat_lon_by_date(mdat, x) for x in mdat['year'].unique()]
assert np.sum(prism_check) == len(mdat['year'].unique())



# Merge NLCD
ldat = pd.read_csv(f"data/{waterbasin}/aso_nlcd_lookup.csv")
ldat = ldat.assign(lat = np.round(ldat['lat'], 4),
                         lon = np.round(ldat['lon'], 4))
ldat = ldat.assign(lat_lon = ldat['lat'].astype(str) + "_" + ldat['lon'].astype(str))
ldat = ldat.drop_duplicates(subset='lat_lon')
ldat = ldat[['lat_lon', 'nlcd_grid']]

mdat = mdat.merge(ldat, on='lat_lon', how='left')

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

# [check_unique_lat_lon_by_date(mdat1, x) for x in mdat1['year'].unique()]

# Merge NLCD
mdat1 = mdat1.drop(columns=['lat', 'lon'])
mdat1 = mdat1.merge(ldat_2013, left_on='nlcd_grid', right_on='lat_lon', how='left')

mdat1 = mdat1.rename(columns={'lat_lon_x': 'lat_lon'})
mdat1 = mdat1.drop(columns=['lat_lon_y'])

# [check_unique_lat_lon_by_date(mdat1, x) for x in mdat1['year'].unique()]

# Model data 2016, 2017
mdat2 = mdat[(mdat['year'] >= 2014) & (mdat['year'] <= 2017)]

# Merge NLCD
mdat2 = mdat2.drop(columns=['lat', 'lon'])
mdat2 = mdat2.merge(ldat_2016, left_on='nlcd_grid', right_on='lat_lon', how='left')

mdat2 = mdat2.rename(columns={'lat_lon_x': 'lat_lon'})
mdat2 = mdat2.drop(columns=['lat_lon_y'])

# [check_unique_lat_lon_by_date(mdat2, x) for x in mdat2['year'].unique()]

# Model data 2018-2022
mdat3 = mdat[mdat['year'] >= 2018]

# Merge NLCD
mdat3 = mdat3.drop(columns=['lat', 'lon'])
mdat3 = mdat3.merge(ldat_2019, left_on='nlcd_grid', right_on='lat_lon', how='left')

mdat3 = mdat3.rename(columns={'lat_lon_x': 'lat_lon'})
mdat3 = mdat3.drop(columns=['lat_lon_y'])

# [check_unique_lat_lon_by_date(mdat3, x) for x in mdat3['year'].unique()]

mdat_nlcd = pd.concat([mdat1, mdat2, mdat3], axis=0)

mdat_nlcd['lat_x_lon'] = mdat_nlcd['lat'] * mdat_nlcd['lon']

mdat_nlcd = mdat_nlcd.assign(month = pd.to_datetime(mdat_nlcd['date']).dt.strftime("%m"))

print("Checking data merged correctly with NLCD")
nlcd_check = [check_unique_lat_lon_by_date(mdat_nlcd, x) for x in mdat_nlcd['year'].unique()]
assert np.sum(nlcd_check) == len(mdat_nlcd['year'].unique())

# return mdat_nlcd



# Start of function to generate predictions

# Different from data step (removed SWE and Month)
pred_dat = mdat_nlcd[['date', 'lat_lon', 'lat', 'lon', 
       'prism_grid', 'snow', 'tmean', 'tmax', 'tmin', 'ppt', 'gridNumber',
       'aso_date', 'elevation', 'year', 'month',
       'nlcd_grid', 'landcover']]


months = pd.get_dummies(pred_dat['month'])
landcover = pd.get_dummies(pred_dat['landcover'])

# Get month to remove
rm_month = pred_dat['month'].unique()

months_ = ["01", "02", "03", "04", "05", "06", "07", "08"]
months_.remove(rm_month)

for month_ in months_:
    months[f"{month_}"] = 0


# # Add missing months
# months['01'] = 0
# months['03'] = 0
# months['04'] = 0
# months['05'] = 0
# months['06'] = 0
# months['07'] = 0
# months['08'] = 0

pred_dat = pd.concat([pred_dat, months, landcover], axis=1)
pred_dat = pred_dat.drop(columns=['month', 'landcover'])

pred_dat['lat_x_lon'] = pred_dat['lat'] * pred_dat['lon']

pred_dat = pred_dat.assign(doy = pd.to_datetime(pred_dat['date']).dt.strftime("%j"))

pred_dat = pred_dat[['snow',     'tmean',       'ppt',       'doy',       'lat', 'year',
          'lon', 'lat_x_lon', 'elevation',        '01',        '02',
          '03',        '04',        '05',        '06',        '07',
          '08',          11,          12,          21,          22,
           23,          24,          31,          41,          42,
           43,          52,          71,          90,          95]]

pred_dat.columns = pred_dat.columns.astype(str)

pred_dat.to_parquet(f"data/{waterbasin}/pred_data_elevation_prism_sinceSep_nlcd_Apr01_V2.parquet", compression=None)

pred_dat = pd.read_parquet(f"data/{waterbasin}/pred_data_elevation_prism_sinceSep_nlcd_Apr01_V2.parquet")

# Generate predictions
moddat = pred_dat.drop(columns=['year'])

# Save the Keras model
mod = load_model(f"data/{waterbasin}/models/NN-ASO-SWE-Model_V2.h5")

# Save the Scaler object
scaler = joblib.load(f"data/{waterbasin}/models/NN-ASO-SWE-Scaler_V2.pkl")

# Scale data
X_pred_train_scaled = scaler.fit_transform(moddat)

# Use the model for prediction
y_pred = mod.predict(X_pred_train_scaled)

pred_dat_trained = pred_dat.assign(swe_pred = y_pred.ravel())
pred_dat_trained['year'] = pred_dat['year']

pred_dat_trained.to_parquet(f"data/{waterbasin}/predictions/NN-ASO-SWE-2010-2021_Apr01_V2.parquet", compression=None)

pred_dat_trained = pd.read_parquet(f"data/{waterbasin}/predictions/NN-ASO-SWE-2010-2021_Apr01_V2.parquet")


