import glob
import os 
import datetime

import pandas as pd
import geopandas as gpd
import numpy as np

import xarray as xr
import rioxarray
import rasterio
from shapely.geometry import Point



def download_ASO_SD():
    with open("ASO_SD_download_list.txt") as f:
        for line in f:
            # Remove any trailing whitespace (such as a newline)
            url = line.strip()
            filename = url.split("/")[-1]
            print(url)

            # Loop until the file can be loaded with geopandas
            while True:
                os.system(f"wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --auth-no-challenge=on --content-disposition {url} -O data/ASO/SD/{filename}")

                # Try to load the file with geopandas
                try:
                    test = xr.open_rasterio(f"data/ASO/SD/{filename}")
                    break  # Exit the loop if the file can be loaded
                except Exception as e:
                    print(f"{filename} cannot be loaded with geopandas: {str(e)}")
                    os.remove(f"data/ASO/SD/{filename}")  # Remove the file if it can't be loaded
                    continue  # Continue the loop to download the file again


def download_ASO_SWE():
    with open("ASO_SWE_download_list.txt") as f:
        for line in f:
            # Remove any trailing whitespace (such as a newline)
            url = line.strip()
            filename = url.split("/")[-1]
            print(url)

            # Loop until the file can be loaded with geopandas
            while True:
                os.system(f"wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --auth-no-challenge=on --content-disposition {url} -O data/ASO/SWE/{filename}")

                # Try to load the file with geopandas
                try:
                    test = xr.open_rasterio(f"data/ASO/SWE/{filename}")
                    break  # Exit the loop if the file can be loaded
                except Exception as e:
                    print(f"{filename} cannot be loaded with geopandas: {str(e)}")
                    os.remove(f"data/ASO/SWE/{filename}")  # Remove the file if it can't be loaded
                    continue  # Continue the loop to download the file again


def download_ASO_SWE_new():
    with open("ASO_SWE_download_list_2020-2023.txt") as f:
        for line in f:
            # Remove any trailing whitespace (such as a newline)
            url = line.strip()
            filename = url.split("/")[-1]
            print(url)

            # Loop until the file can be loaded with geopandas
            os.system(f"wget {url} -O data/temp/aso.zip")
            os.system("unzip -j data/temp/aso.zip -d data/temp/.")
            os.system("cp data/temp/*swe*.tif data/ASO/SWE/.")
            os.system("rm data/temp/*")
            # Try to load the file with geopandas
            # try:
            #     test = xr.open_rasterio(f"data/ASO/SWE/{filename}")
            #     break  # Exit the loop if the file can be loaded
            # except Exception as e:
            #     print(f"{filename} cannot be loaded with geopandas: {str(e)}")
            #     os.remove(f"data/ASO/SWE/{filename}")  # Remove the file if it can't be loaded
            #     continue  # Continue the loop to download the file again


def proc_ASO_SD():
    files = glob.glob("data/ASO/SD/*.tif")
    # file_ = files[0]

    # file_ = "data/ASO/ASO_50M_SD_USCOCB_20160404.tif"
    outlist = []
    for file_ in files:
        print(f"Processing: {file_}")
        ds = rioxarray.open_rasterio(file_)
        ds = ds.rio.reproject("EPSG:4326")

        # Get date
        date = file_.split("/")[-1].split("_")[-1].replace(".tif", "")
        date = date[0:4] + "-" + date[4:6] + "-" + date[6:8]

        # Get site
        site = file_.split("/")[-1].split("_")[-2]

        dat = ds.to_dataframe("SD").reset_index()
        dat = dat[dat['SD'] != -9999.0]

        dat = dat[['x', 'y', 'SD']]
        dat.columns = ['lon', 'lat', 'SD']

        dat.insert(0, 'date', date)
        dat.insert(1, 'site', site)

        outdat = dat.reset_index(drop=True)
        outlist.append(outdat)


    savedat = pd.concat(outlist)
    savedat.to_csv("data/processed/ASO-SD-2013-2019.csv", index=False)


def proc_ASO_SWE_shp(gdf):
    files = glob.glob("data/ASO/SWE/*.tif")

    outlist = []
    for file_ in files:
        print(f"Processing: {file_}")
        ds = rioxarray.open_rasterio(file_)
        ds = ds.rio.reproject("EPSG:4326")

        # Setup missing values
        ds.attrs['_FillValue'] = np.nan
        ds = ds.where(ds != -9999.0, np.nan)

        try: 
            dat = ds.rio.clip(gdf.geometry, all_touched=True, drop=True, invert=False, from_disk=True) 
            dat = dat.to_series().reset_index()
            dat.columns = ['band', 'y', 'x', 'SWE']

            if "ASO_Tuolumne" in file_:
                # Get date
                sub_date = file_.split("/")[-1].split("_")[-3]
                year = file_.split("/")[-1].split("_")[-3][0:4]
                month = file_.split("/")[-1].split("_")[-3][4:7]
                day = file_.split("/")[-1].split("_")[-3][7:9]
                date_string = year + "-" + month + "-" + day

                # Convert the date object to the desired format using strftime()
                date_object = datetime.datetime.strptime(date_string, "%Y-%b-%d")
                date = date_object.strftime("%Y-%m-%d")

                # Get site
                site = file_.split("/")[-1].split("_")[2]

                if site == sub_date:
                    site = file_.split("/")[-1].split("_")[1]                    

            elif "Blue" or "TenMileCk" in file_:
                print(file_)
                date = file_.split("_")[-3]
                date = date[0:10]
                
                if len(date) == 9:
                    date_object = datetime.datetime.strptime(date, "%Y%b%d")
                elif len(date) == 10:
                    date_object = datetime.datetime.strptime(date, "%Y%B%d")

                date = date_object.strftime("%Y-%m-%d")
                site = file_.split("_")[-4]

            else:

                # Get date
                date = file_.split("/")[-1].split("_")[-1].replace(".tif", "")
                date = date[0:4] + "-" + date[4:6] + "-" + date[6:8]

                # Get site
                site = file_.split("/")[-1].split("_")[-2]

            dat = dat[['x', 'y', 'SWE']]
            dat.columns = ['lon', 'lat', 'SWE']

            dat.insert(0, 'date', date)
            dat.insert(1, 'site', site)

            outdat = dat.reset_index(drop=True)
            outlist.append(outdat)
        except Exception as e:
            print(e)

    outdat = pd.concat(outlist)
    outdat['date'].unique()

    return outdat





if __name__ == "__main__":
    
    download_ASO_SD()

    download_ASO_SWE()
    
    download_ASO_SWE_new()







