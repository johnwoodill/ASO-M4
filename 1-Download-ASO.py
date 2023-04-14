import glob
import xarray as xr
import rioxarray
import os 
import rasterio
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd


def download_ASO():
    with open("ASO_download_list.txt") as f:
        for line in f:
            # Remove any trailing whitespace (such as a newline)
            url = line.strip()
            filename = url.split("/")[-1]
            print(url)

            # Loop until the file can be loaded with geopandas
            while True:
                os.system(f"wget {url} -O data/ASO/{filename}")

                # Try to load the file with geopandas
                try:
                    test = xr.open_rasterio(f"data/ASO/{filename}")
                    break  # Exit the loop if the file can be loaded
                except Exception as e:
                    print(f"{filename} cannot be loaded with geopandas: {str(e)}")
                    os.remove(f"data/ASO/{filename}")  # Remove the file if it can't be loaded
                    continue  # Continue the loop to download the file again




def proc_ASO():
    files = glob.glob("data/ASO/*.tif")
    # file_ = files[0]

    # file_ = "ASO_50M_SD_USCOCB_20160404.tif"
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

        dat = ds.to_dataframe("SWE").reset_index()
        dat = dat[dat['SWE'] != -9999.0]

        dat = dat[['x', 'y', 'SWE']]
        dat.columns = ['lon', 'lat', 'SWE']

        dat.insert(0, 'date', date)
        dat.insert(1, 'site', site)

        outdat = dat.reset_index(drop=True)
        outlist.append(outdat)


    savedat = pd.concat(outlist)
    savedat.to_csv("data/processed/ASO-2013-2019.csv", index=False)




if __name__ == "__main__":
    
    download_ASO()
    
    proc_ASO()








