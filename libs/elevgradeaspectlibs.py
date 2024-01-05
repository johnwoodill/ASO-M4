import requests
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray
import numpy as np

from sentinelhub import SentinelHubRequest, DataCollection, MimeType, CRS, BBox

from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session

import math
import os

import pandas as pd
import geopandas as gpd
import rioxarray as rxr
import xarray as xr
import numpy as np
import multiprocessing
from tqdm import tqdm
from math import radians, cos, sin, asin, sqrt

tqdm.pandas()



from sentinelhub import (
    SHConfig,
    CRS,
    BBox,
    DataCollection,
    DownloadRequest,
    MimeType,
    # MosaickingOrder,
    SentinelHubDownloadClient,
    SentinelHubRequest,
    bbox_to_dimensions,
)


# Sentinel Hub login
instance_id = '8cb2a420-91fb-45fe-aa21-46e825d7620d'
ID = 'e843668a-2670-4e2b-b881-7905ee9ce095'
SECRET = '0USn166rzrQp1qFvNaLDP58v50DD1QSm'

# Create a session
client = BackendApplicationClient(client_id=ID)
oauth = OAuth2Session(client=client)

# Get token for the session
token = oauth.fetch_token(token_url='https://services.sentinel-hub.com/auth/realms/main/protocol/openid-connect/token',
                          client_secret=SECRET, include_client_id=True)

# All requests using this session will have an access token automatically added
resp = oauth.get("https://services.sentinel-hub.com/configuration/v1/wms/instances")
print(resp.content)



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





def proc_grade_elev_watershed(basin_name, fp):

    # Assuming bounds are defined as in your provided snippet
    bounds = fp.geometry.bounds
    lon1 = np.round(bounds['minx'][0], 1) - 0.1
    lat1 = np.round(bounds['miny'][0], 1) - 0.1
    lon2 = np.round(bounds['maxx'][0], 1) + 0.1
    lat2 = np.round(bounds['maxy'][0], 1) + 0.1

    # Define the number of rows and columns for the grid
    rows = 4
    cols = 5

    # Calculate the width and height of each section
    width = (lon2 - lon1) / cols
    height = (lat2 - lat1) / rows

    # Initialize a list to hold the sections
    quadrants = []

    # Loop through each row and column to create the sections
    for i in range(rows):
        for j in range(cols):
            minx = lon1 + j * width
            maxx = lon1 + (j + 1) * width
            miny = lat1 + i * height
            maxy = lat1 + (i + 1) * height
            section = {'minx': np.round(minx, 4), 'miny': np.round(miny, 4), 'maxx': np.round(maxx, 4), 'maxy': np.round(maxy, 4)}
            quadrants.append(section)


    lst_ = []
    for i in range(len(quadrants)):
        print(f"Processing: {i}")
        lon1 = np.round(quadrants[i]['minx'], 1)
        lat1 = np.round(quadrants[i]['miny'], 1)
        lon2 = np.round(quadrants[i]['maxx'], 1)
        lat2 = np.round(quadrants[i]['maxy'], 1)

        access_token = token['access_token']

        # API endpoint for the Sentinel Hub process API
        api_url = 'https://services.sentinel-hub.com/api/v1/process'

        # Headers for the request
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json'
        }

        # The payload for the request, formatted as a Python dictionary
        payload = {
            "input": {
                "bounds": {
                    "properties": {
                        "crs": "http://www.opengis.net/def/crs/EPSG/0/4326"
                    },
                    "bbox": [
                        lon1,
                        lat1,
                        lon2,
                        lat2
                    ]
                },
                "data": [{
                    "type": "dem",
                    "dataFilter": {
                        "demInstance": "COPERNICUS_30"
                    },
                    "processing": {
                        "upsampling": "BILINEAR",
                        "downsampling": "BILINEAR"
                    }
                }]
            },
            "output": {
                "resx": 0.0003,
                "resy": 0.0003,
                "responses": [{
                    "identifier": "default",
                    "format": {
                        "type": "image/tiff"
                    }
                }]
            },
            "evalscript": """
            //VERSION=3
            function setup() {
              return {
                input: ["DEM"],
                output: { bands: 1 ,
                          sampleType: SampleType.FLOAT32}
              }
            }
            function evaluatePixel(sample) {
              return [sample.DEM]
            }
            """
        }

        # Make the POST request
        response = requests.post(api_url, headers=headers, json=payload)

        # Check if the request was successful
        if response.status_code == 200:
            # If successful, write the content to a file
            with open('/tmp/output.tiff', 'wb') as file:
                file.write(response.content)
            print("Download successful, file saved!")
        else:
            # If not successful, print the error
            print(f"Download failed: {response.status_code}")
            print(response.text)

        xdat = xr.open_dataset("/tmp/output.tiff")
        # xdat = xdat.rio.reproject("EPSG:32611")

        xdat = xdat.rio.reproject("EPSG:4326")    
        xdat = get_aspect_grade(xdat)
        
        xdat = xdat.to_dataframe().reset_index()
        xdat = xdat[['x', 'y', 'band_data', 'slope', 'aspect']]
        xdat.columns = ['lon', 'lat', 'elev_m', 'slope', 'aspect']
        lst_.append(xdat)

    # Bin and save
    outdat = pd.concat(lst_)
    outdat.to_csv(f"data/{basin_name}/elev_grade_aspect/elev_grade_aspect.csv", index=False)
    print(f"Saved: data/{basin_name}/elev_grade_aspect/elev_grade_aspect.csv")


def get_aspect_grade(xdat):
    # Assuming band_data is your elevation
    elevation = xdat['band_data'].isel(band=0)  # Selecting the first band if multiple
    dz_dx, dz_dy = np.gradient(elevation)  # Calculate gradients

    # Calculate slope
    slope = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)) * (180/np.pi)  # Convert to degrees

    # Calculate aspect
    aspect = np.arctan2(-dz_dy, -dz_dx) * (180/np.pi)  # Convert to degrees
    aspect = np.where(aspect < 0, 90 - aspect, 270 - aspect)  # Adjusting the aspect values

    # Convert the numpy arrays to xarray DataArrays
    slope_da = xr.DataArray(slope, dims=elevation.dims, coords=elevation.coords)
    aspect_da = xr.DataArray(aspect, dims=elevation.dims, coords=elevation.coords)

    # Add slope and aspect to the original dataset
    xdat['slope'] = slope_da
    xdat['aspect'] = aspect_da

    return xdat



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


    