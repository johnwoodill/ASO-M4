import pandas as pd
import numpy as np 
import pandas as pd
import os
import wget
import zipfile
import gdal as gdal
import multiprocessing
from dask.delayed import delayed
from dask import compute
import dask.dataframe as dd
import glob
import subprocess
from math import radians, cos, sin, asin, sqrt
from shapely.geometry import Polygon
import rioxarray as rxr
import xarray as xr
import geopandas as gpd



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


# Get PRISM data
def proc_PRISMbil(ThisCol, ThisRow, bil_path):
    """
    Args:
        ThisCol: column from raster file to parse
        ThisRow: row from raster file to parse
        bil_path: path to raster file
    
    Returns:
        Parsed grid_id, X, Y, PRISM data
    """
    # Raster file setup
    ncol = 1405
    nrow = 621

    ds       = gdal.OpenShared(bil_path)
    GeoTrans = ds.GetGeoTransform()
    ColRange = range(ds.RasterXSize)
    RowRange = range(ds.RasterYSize)
    rBand    = ds.GetRasterBand(1) # first band
    nData    = rBand.GetNoDataValue()
    if nData == None:
        nData = -9999 # set it to something if not set

    # specify the centre offset
    HalfX    = GeoTrans[1] / 2
    HalfY    = GeoTrans[5] / 2

    # Get parsed data
    RowData = rBand.ReadAsArray(0, ThisRow, ds.RasterXSize, 1)[0]
    if RowData[ThisCol] >= -100:   # Greater than minTemp -100 C
        X = GeoTrans[0] + ( ThisCol * GeoTrans[1] )
        Y = GeoTrans[3] + ( ThisRow * GeoTrans[5] )
        X += HalfX
        Y += HalfY
        grid_id = ((ThisRow-1)*ncol)+ThisCol
        return (grid_id, X, Y, RowData[ThisCol])
    else:
        return (-9999, -9999, -9999, -9999)


def proc_prism_files(bil_path, gdf, timestep):
    """
    Args:
        bil_path: path to raster file
        gdf: shapefile to clip
        timestep: "monthly" or "daily"
    
    Returns:
        Nothing
    """
    # Process raster
    ds       = gdal.OpenShared(bil_path)
    ColRange = range(ds.RasterXSize)
    RowRange = range(ds.RasterYSize)

    # List compress grids and data
    lst = [proc_PRISMbil(x, y, bil_path) for x in ColRange for y in RowRange]

    # Build df of lst compression
    gridNumber = [x[0] for x in lst]
    lon = [x[1] for x in lst]
    lat = [x[2] for x in lst]
    value = [x[3] for x in lst]
    # prism_date =  bil_path[-16:-8]    
    prism_date = bil_path.split("_")[-2]
    prism_var = bil_path.split("_")[1]

    outdat = pd.DataFrame({'date': prism_date, 'gridNumber': gridNumber, 'var': prism_var, 'value': value, 'lon': lon, 'lat': lat})
    
    # Remove NA
    outdat = outdat[outdat['gridNumber'] != -9999].sort_values(['lon'], ascending=[False])
    outdat = outdat[outdat['gridNumber'] != -9999].sort_values(['lat'], ascending=[False])
    outdat = outdat[['date', 'gridNumber', 'var', 'value', 'lon', 'lat']].reset_index(drop=True)

    # Get save parameters
    prism_var = outdat['var'].iat[0]
    prism_date = outdat['date'].iat[0]

    # Filter dataframe from shapefile
    grid_outdat = outdat[['date', 'var', 'gridNumber', 'value', 'lon', 'lat']]
    grid_outdat.columns = ['date', 'var', 'gridNumber', 'value', 'longitude', 'latitude']

    #Build GeoDataFrame to clip
    xdat = pd.DataFrame(grid_outdat)
    edgar_gdf = gpd.GeoDataFrame(xdat, geometry=gpd.points_from_xy(xdat['longitude'], xdat['latitude']),
        crs="EPSG:4326")

    # Clip to shapefile
    indat = gpd.clip(edgar_gdf, gdf)

    # Clean up
    grid_dat = indat[['date', 'longitude', 'latitude', 'gridNumber', 'var', 'value']].reset_index(drop=True)

    # Filter nearest farms lat and long and save
    print(f"Saving: data/prism_output/{timestep}/prism_{prism_date}_{prism_var}.csv")
    grid_dat.to_csv(f"data/prism_output/{timestep}/prism_{prism_date}_{prism_var}.csv", index=False)

    return lst


def proc_prism(dat, gdf, timestep, par=True):
    """
    Args:
        dat: groupby data from
        gdf: shapefile to clip
        timestep: "monthly", or "daily"
        par: parallel True or False
    
    Returns:
        0
    """
    if timestep == "daily":
        res = "4kmD2"
    elif timestep == "monthly":
        res = "4kmM3"

    # Get save parameters
    prism_var = dat['var'].iat[0]
    prism_year = dat['year'].iat[0]

    print(f"Processing: {prism_var}-{prism_year}")

    # Get downloaded zip files
    file_path = sorted(glob.glob(f"data/prism.nacse.org/{timestep}/{prism_var}/{prism_year}/*_bil.zip"))
    
    # Unzip
    for file_ in file_path:
        subprocess.call(['unzip', '-o', file_, '-d', '/tmp/dump/'], stdout=subprocess.DEVNULL) 

    # Get *.bil file from zip    
    files_ = glob.glob(f"/tmp/dump/PRISM_{prism_var}_stable_{res}_{prism_year}*_bil.bil")

    # Processes in parallel or single
    if par == True:
        compute([delayed(proc_prism_files)(file_, gdf, timestep) for file_ in files_], scheduler='processes')
    else:
        [proc_prism_files(file_, gdf, timestep) for file_ in files_]

    # Trash collecting
    print("Trashing cleaning")
    del_files = glob.glob(f"/tmp/dump/PRISM_{prism_var}_stable_4kmD2_{prism_year}*")
    for filename in del_files:
        os.remove(filename)
    return 0


def proc_daily(gdf, location, min_year=1981, max_year=2023):
    
    # Build dataframe to process years and variables
    year_ = np.arange(min_year, max_year, 1).astype(str)
    var_ = ["tmax", "tmin", "ppt"]

    lst_ = [(x, y) for x in year_ for y in var_]
    year = [x[0] for x in lst_]
    var = [x[1] for x in lst_]
    indf = pd.DataFrame({'year': year, 'var': var}).reset_index()

    # Get groups
    gb = indf.groupby('index')
    gb_i = [gb.get_group(x) for x in gb.groups]

    # Process
    [proc_prism(x, gdf, "daily", par=True) for x in gb_i]

    # Bin up daily data
    prism_files = glob.glob('data/prism_output/daily/*.csv')
    len_prism_files = len(prism_files)

    outdat = pd.DataFrame()
    df_list = []
    for x, file_ in enumerate(prism_files):
        print(np.round((x/len_prism_files) * 100))
        # print(file_)
        df_list.append(pd.read_csv(file_))
    
    # Concat and save
    outdat = pd.concat(df_list)
    outdat.to_csv(f"data/{location}/processed/{location}_PRISM_daily_{min_year}-{max_year}.csv", index=False)


if __name__ == "__main__":

    # shape_loc = "data/shapefiles/tuolumne_watershed/Tuolumne_Watershed.shp"
    # gdf = gpd.read_file(shape_loc)
    # gdf = gdf.to_crs(epsg=4326)

    # min_year = 2015
    # max_year = 2016

    # proc_daily(gdf, "Tuolumne_Watershed")






