import pandas as pd 
import numpy as np
import os
import urllib
import requests
import urllib.request
import time
import multiprocessing
from datetime import datetime, timedelta

import geopandas as gpd
import rasterio
import xarray as xr
import rioxarray as rxr
from shapely.geometry import Point

import dask.dataframe as dd
from dask import delayed
from dask.diagnostics import ProgressBar

from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

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


def proc_prediction(waterbasin, shape_loc, pred_dates):
    decimal_point = 4
    gdf = gpd.read_file(shape_loc)
    gdf = gdf.to_crs(epsg=4326)
    # gdf = gdf.set_crs("EPSG:4326")  # Replace EPSG:4326 with your CRS

    # Get ASO Data
    # aso_dat = proc_ASO_SWE_shp(gdf)
    # aso_dat = aso_dat.dropna()
    # aso_dat = aso_dat[aso_dat['SWE'] >= 0]

    # aso_dat = aso_dat.assign(lat = np.round(aso_dat['lat'], 4),
    #                          lon = np.round(aso_dat['lon'], 4))

    # # Different from data step
    # aso_dat['lat_lon'] = aso_dat['lat'].astype(str) + "_" + aso_dat['lon'].astype(str)

    aso_dat = pd.read_csv(f"data/{watershed}/processed/aso_basin_data.csv")

    aso_dat = aso_dat.drop_duplicates(subset=['lat_lon'])
    aso_dat = aso_dat[['lat_lon', 'lat', 'lon']]

    len(aso_dat)    
    len(aso_dat.dropna())/len(aso_dat)

    # Differernt from data step
    aso_dat['key'] = 1
    pred_dates['key'] = 1

    aso_dat = aso_dat.merge(pred_dates, on='key').drop('key', axis=1)
    aso_dat['year'].unique()

    # Get elevation data
    aso_elev = pd.read_csv(f"data/{watershed}/processed/aso_elev_grade_aspect.csv")
    aso_elev.columns = ['lat', 'lon', 'lat_lon', 'elevation', 'slope', 'aspect']
    # aso_elev = pd.read_parquet(f"data/{watershed}/processed/aso_elevation.parquet")
    # aso_elev.columns = ['lat', 'lon', 'elevation']

    # aso_elev = aso_elev.assign(lat = np.round(aso_elev['lat'], 4),
    #                          lon = np.round(aso_elev['lon'], 4))

    # aso_elev['lat_lon'] = aso_elev['lat'].apply(lambda x: f"{x:.{decimal_point}f}") + "_" + aso_elev['lon'].apply(lambda x: f"{x:.{decimal_point}f}")

    # aso_elev = aso_elev.assign(lat_lon = aso_elev['lat'].astype(str) + "_" + aso_elev['lon'].astype(str))
    aso_elev = aso_elev[['lat_lon', 'elevation', 'slope', 'aspect']]
    # aso_elev = aso_elev.drop_duplicates(subset=['lat_lon'])

    # Prism data
    prism_filename = glob.glob(f"data/{watershed}/processed/*PRISM*")
    prism_dat = pd.read_csv(prism_filename[0])

    # Convert the 'date' column to a datetime object just once
    prism_dat['date'] = pd.to_datetime(prism_dat['date'], format="%Y%m%d")

    # Extract the month and year using vectorized operations
    prism_dat['month'] = prism_dat['date'].dt.month  # This will already be integer
    prism_dat['year'] = prism_dat['date'].dt.year  

    # prism_dat = prism_dat.assign(date = pd.to_datetime(prism_dat['date'], format = "%Y%m%d"))
    # prism_dat = prism_dat.assign(month = pd.to_datetime(prism_dat['date']).dt.strftime("%m"))
    # prism_dat = prism_dat.assign(year = pd.to_datetime(prism_dat['date']).dt.strftime("%Y"))

    # prism_dat = prism_dat.assign(month = prism_dat['month'].astype(int),
    #                              year = prism_dat['year'].astype(int))

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
    aso_prism_lookup = pd.read_csv(f"data/{watershed}/processed/aso_prism_lookup.csv")
    aso_prism_lookup['lat_lon'] = aso_prism_lookup['lat'].apply(lambda x: f"{x:.{decimal_point}f}") + "_" + aso_prism_lookup['lon'].apply(lambda x: f"{x:.{decimal_point}f}")

    # aso_prism_lookup = aso_prism_lookup.assign(lat_lon = aso_prism_lookup['lat'].astype(str) + "_" + aso_prism_lookup['lon'].astype(str))
    aso_prism_lookup = aso_prism_lookup.drop_duplicates(subset=['lat_lon'])
    aso_prism_lookup = aso_prism_lookup[['lat_lon', 'prism_grid']]

    mdat = mdat.merge(aso_prism_lookup, on='lat_lon', how='left')

    mdat = mdat.merge(prism_dat, left_on=['date', 'prism_grid'], right_on=['aso_date', 'gridNumber'], how='left')

    print("Checking data merged correctly with PRISM")
    prism_check = [check_unique_lat_lon_by_date(mdat, x) for x in mdat['year'].unique()]
    assert np.sum(prism_check) == len(mdat['year'].unique())

    # Merge NLCD
    print("Processing NLCD")
    ldat = pd.read_csv(f"data/{watershed}/processed/aso_nlcd_lookup.csv")
    # ldat = ldat.assign(lat = np.round(ldat['lat'], 4),
    #                    lon = np.round(ldat['lon'], 4))
    
    ldat['lat_lon'] = ldat['lat'].apply(lambda x: f"{x:.{decimal_point}f}") + "_" + ldat['lon'].apply(lambda x: f"{x:.{decimal_point}f}")

    # ldat = ldat.assign(lat_lon = ldat['lat'].astype(str) + "_" + ldat['lon'].astype(str))
    ldat = ldat.drop_duplicates(subset='lat_lon')
    ldat = ldat[['lat_lon', 'nlcd_grid']]

    mdat = mdat.merge(ldat, on='lat_lon', how='left')

    ldat_2013 = pd.read_csv(f"data/{watershed}/processed/nlcd_2019_land_cover_l48_20210604.csv")
    ldat_2013 = ldat_2013.assign(lat_lon = ldat_2013['lat'].astype(str) + "_" + ldat_2013['lon'].astype(str))
    ldat_2013 = ldat_2013.drop(columns=['year'])

    ldat_2016 = pd.read_csv(f"data/{watershed}/processed/nlcd_2016_land_cover_l48_20210604.csv")
    ldat_2016 = ldat_2016.assign(lat_lon = ldat_2016['lat'].astype(str) + "_" + ldat_2016['lon'].astype(str))
    ldat_2016 = ldat_2016.drop(columns=['year'])

    ldat_2019 = pd.read_csv(f"data/{watershed}/processed/nlcd_2019_land_cover_l48_20210604.csv")
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
    assert len(mdat_nlcd.dropna()) == len(mdat_nlcd)
    assert len(mdat_nlcd.drop_duplicates()) == len(mdat_nlcd)

    return mdat_nlcd


def gen_predictions(proc_pred_dat):
    print("Generating Model Data Frame")
    min_year = proc_pred_dat['year'].min()
    max_year = proc_pred_dat['year'].max()

    month = proc_pred_dat['month'].iat[0]
    formatted_month = datetime.datetime.strptime(month, "%m").strftime("%b01")

    # Get common column names
    pred_dat = proc_pred_dat[['date', 'lat_lon', 'lat', 'lon', 
           'prism_grid', 'snow', 'tmean', 'tmax', 'tmin', 'ppt', 'gridNumber',
           'aso_date', 'elevation', 'slope', 'aspect', 'year', 'month',
           'nlcd_grid', 'landcover']]

    # Generate controls
    months = pd.get_dummies(pred_dat['month'], prefix='month')
    landcover = pd.get_dummies(pred_dat['landcover'], prefix='landcover')

    # Get month to remove
    rm_month = pred_dat['month'].unique()

    # Control for other months not in columns
    months_ = ["month_01", "month_02", "month_03", "month_04", "month_05", 
               "month_06", "month_07", "month_08"]

    months_.remove(f"month_{rm_month[0]}")

    for month_ in months_:
        months[f"{month_}"] = 0

    pred_dat = pd.concat([pred_dat, months, landcover], axis=1)
    pred_dat = pred_dat.drop(columns=['month', 'landcover'])

    # Interaction of lat/lon
    pred_dat['lat_x_lon'] = pred_dat['lat'] * pred_dat['lon']

    # Get day of year
    pred_dat = pred_dat.assign(doy = pd.to_datetime(pred_dat['date']).dt.strftime("%j"))

    # Load column names from model
    col_names = np.load(f"data/{watershed}/models/col_list.npy")

    pred_dat = pred_dat[col_names]

    # pred_dat = pred_dat[['snow',     'tmean',       'ppt',       'doy',       'lat', 'year',
    #           'lon', 'lat_x_lon', 'elevation',        '01',        '02',
    #           '03',        '04',        '05',        '06',        '07',
    #           '08',          11,          12,          21,          22,
    #            23,          24,          31,          41,          42,
    #            43,          52,          71,          90,          95]]

    pred_dat.columns = pred_dat.columns.astype(str)

    print(f"Saving: data/{watershed}/pred_data_elevation_prism_sinceSep_nlcd_{formatted_month}_{min_year}_{max_year}.parquet")
    pred_dat.to_parquet(f"data/{watershed}/pred_data_elevation_prism_sinceSep_nlcd_{formatted_month}_{min_year}_{max_year}.parquet", compression=None)

    # Generate predictions
    # moddat = pred_dat.drop(columns=['year'])

    # Save the Keras model
    print("Loading Keras model for predictions")
    mod = load_model(f"data/{watershed}/models/NN-ASO-SWE-Model.h5")

    # Save the Scaler object
    scaler = joblib.load(f"data/{watershed}/models/NN-ASO-SWE-Scaler.pkl")

    # Scale data
    X_pred_train_scaled = scaler.fit_transform(pred_dat)

    # Use the model for prediction
    y_pred = mod.predict(X_pred_train_scaled)

    y_pred = np.exp(y_pred) - 1

    pred_dat_trained = pred_dat.assign(swe_pred = y_pred.ravel())

    pred_dat_trained['year'] = proc_pred_dat['year']

    print(f"Saved: data/{watershed}/predictions/NN-ASO-SWE-{formatted_month}_{min_year}_{max_year}.parquet")
    pred_dat_trained.to_parquet(f"data/{watershed}/predictions/NN-ASO-SWE-{formatted_month}_{min_year}_{max_year}.parquet", compression=None)



if __name__ == "__main__":

    decimal_point = 4

    watershed = "Tuolumne_Watershed"
    shape_loc = glob.glob(f"data/{watershed}/shapefiles/*.shp")[0]

    pred_dates = generate_dates(1981, 2021, 4, 1)
    
    mdat_nlcd = proc_prediction(watershed, shape_loc, pred_dates)

    output = gen_predictions(mdat_nlcd)

    # ----------------------------------------------------------

    watershed = "Blue_Dillon_Watershed"
    shape_loc = glob.glob(f"data/{watershed}/shapefiles/*.shp")[0]

    pred_dates = generate_dates(1981, 2022, 4, 1)
    
    mdat_nlcd = proc_prediction(watershed, shape_loc, pred_dates)

    output = gen_predictions(mdat_nlcd)

    # ----------------------------------------------------------

    watershed = "Dolores_Watershed"
    shape_loc = glob.glob(f"data/{watershed}/shapefiles/*.shp")[0]
    
    pred_dates = generate_dates(1981, 2022, 4, 1)

    mdat_nlcd = proc_prediction(watershed, shape_loc, pred_dates)

    output = gen_predictions(mdat_nlcd)

    # ----------------------------------------------------------

    watershed = "Conejos_Watershed"
    shape_loc = glob.glob(f"data/{watershed}/shapefiles/*.shp")[0]
    
    pred_dates = generate_dates(1981, 2022, 4, 1)

    mdat_nlcd = proc_prediction(watershed, shape_loc, pred_dates)

    output = gen_predictions(mdat_nlcd)


