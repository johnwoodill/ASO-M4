import pandas as pd
import geopandas as gpd
import rioxarray as rxr
import xarray as xr
import numpy as np
import multiprocessing
from tqdm import tqdm

tqdm.pandas()


def find_closest_grid_nlcd_parallel(row):
    return find_closest_grid(row['lat'], row['lon'], ndat, 'lat_lon', decimal=0.001)


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



def get_nlcd(filename, shape_loc, basin_name):

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



def proc_nlcd(gdf, shape_loc, basin_name):
    # ----------------------------------
    # Land cover
    # nlcd_files  = ["nlcd_2019_land_cover_l48_20210604.img",
    #                "nlcd_2016_land_cover_l48_20210604.img",
    #                "nlcd_2013_land_cover_l48_20210604.img"]


    # [get_nlcd(x, shape_loc, basin_name) for x in nlcd_files]

    ldat = pd.read_csv(f"data/{basin_name}/processed/aso_basin_data.csv")
    ldat = ldat.drop_duplicates(subset='lat_lon')
    ldat = ldat[['lon', 'lat']].reset_index(drop=True)
    
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

