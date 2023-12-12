import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import os
import requests
from io import StringIO
import pandas as pd

import ulmo   # snotel data

from libs.asolibs import *
from libs.gagelibs import StreamGauge
import glob


def get_nrcs_snotel():
    sno_file = f'data/NRCS_Snow_Station/TuolumneSWE_041981_042023.txt'
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
    nrdat = sno_data
    nrdat = nrdat.assign(year = pd.to_datetime(nrdat['Date']).dt.year)

    # filter for continuous time series & shorten column names
    nrdat = nrdat.drop(columns=incomplete_colms)

    nrdat = nrdat.rename(columns=dict(
        zip(
            nrdat.columns,
            map(
                lambda x: x.replace("Snow Water Equivalent (in) Start of Month Values", "SWE (in)"), nrdat.columns
            )
        )
    ))

    # save processed data
    # nrdat.to_csv(f"data/{dir_name}/{dir_name}_NRCS_SNOTEL_data.csv", index = False)

    nrdat = nrdat.drop(columns=['Date'])
    nrdat.insert(0, 'year', nrdat.pop('year'))

    cols = nrdat.columns
    new_cols = [x.split(" ")[0:2] for x in cols[1:]]
    new_cols = ["_".join(x) for x in new_cols]

    new_cols.insert(0, 'year')

    nrdat.columns = new_cols

    return nrdat


def filter_snotel(gdf, sites_df):

    sites_ = []
    for i in range(len(sites_df)):

        lat = float(sites_df['location'][i]['latitude'])
        lon = float(sites_df['location'][i]['longitude'])
        point = Point(lon, lat)

        point_in = gdf.geometry.contains(point)
        if point_in[0] == True:
            insite = sites_df['code'].iat[i]
            sites_.append(insite)
            print(f"Found point: {insite}")
    return pd.Series(sites_)


def proc_snotel(snotelCode, month=4):

    url = f"https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/" \
          f"customMultiTimeSeriesGroupByStationReport/monthly/start_of_period/" \
          f"{snotelCode}%7Cid=%22%22%7Cname/POR_BEGIN,POR_END/stationId,name,WTEQ::value"

    response = requests.get(url)

    if response.status_code == 200:
        # The data is available in the response's content
        data = response.content

        data_string = data.decode('utf-8')

        # Split the data into lines
        lines = data_string.strip().split("\n")

        # Filter out lines that start with "#"
        data_lines = [line for line in lines if not line.startswith("#")]

        # Join the data lines into a single string
        data_string = "\n".join(data_lines)

        # Create a DataFrame from the data
        df = pd.read_csv(StringIO(data_string), parse_dates=["Date"])

        col_name = df.columns[1].split(" ")[0]

        df.columns = ['date', f'{col_name}_swe']

        df['month'] = pd.to_datetime(df['date']).dt.strftime("%m").astype(int)
        df['year'] = pd.to_datetime(df['date']).dt.strftime("%Y").astype(int)

        df = df[df['month'] == month]

        df = df.iloc[:, [0, 3, 2, 1]]

        return df

    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}")


def get_flowrate(stationID, start_date, end_date):
    url = f"https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/" \
          f"customMultiTimeSeriesGroupByStationReport/monthly/start_of_period/" \
          f"{stationID}%7Cid=%22%22%7Cname/{start_date},{end_date}:M%7C4,M%7C5," \
          f"M%7C6,M%7C7/stationId,name,SRVO::value"

    response = requests.get(url)

    if response.status_code == 200:
        # The data is available in the response's content
        data = response.content

        data_string = data.decode('utf-8')

        # Split the data into lines
        lines = data_string.strip().split("\n")

        # Filter out lines that start with "#"
        data_lines = [line for line in lines if not line.startswith("#")]

        # Join the data lines into a single string
        data_string = "\n".join(data_lines)

        # Create a DataFrame from the data
        df = pd.read_csv(StringIO(data_string), parse_dates=["Date"])

        col_name = df.columns[1].split(" ")[0]

        df.columns = ['date', 'kaf']

        df['month'] = pd.to_datetime(df['date']).dt.strftime("%m").astype(int)
        df['year'] = pd.to_datetime(df['date']).dt.strftime("%Y").astype(int)

        df = df.iloc[:, [0, 3, 2, 1]]

        df = df[(df['month'] >= 4) & (df['month'] <= 7)]

        df = df.groupby('year').agg({'kaf': np.sum}).reset_index()

        df['kaf'] = df['kaf']/1000
        
        return df

    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}")


def get_prism(watershed, min_year, max_year):
    prism_loc = glob.glob(f"data/{watershed}/processed/*PRISM*.csv")[0]
    prism_dat = pd.read_csv(prism_loc)
    prism_dat = prism_dat.assign(date = pd.to_datetime(prism_dat['date'], format = "%Y%m%d"))
    prism_dat = prism_dat.assign(month = pd.to_datetime(prism_dat['date']).dt.strftime("%m"))
    prism_dat = prism_dat.assign(year = pd.to_datetime(prism_dat['date']).dt.strftime("%Y"))

    prism_dat = prism_dat.assign(month = prism_dat['month'].astype(int),
                                 year = prism_dat['year'].astype(int))

    prism_dat = prism_dat[(prism_dat['year'] >= min_year) & prism_dat['year'] <= max_year]
    pivoted_df = prism_dat.pivot_table(index=['date', 'longitude', 'latitude', 'gridNumber', 'month', 'year'],
                                columns='var',
                                values='value').reset_index()

    # Subset proportion of year
    dat1 = pivoted_df[(pivoted_df['month'] >= 10)]
    dat2 = pivoted_df[pivoted_df['month'] <= 3]
    tdat = pd.concat([dat1, dat2])

    tdat['adj_year'] = np.where(tdat['month'] >= 10, tdat['year'] + 1, tdat['year'])
    tdat['tmean'] = (tdat['tmax'] + tdat['tmin']) / 2

    tdat = tdat.groupby(['adj_year']).agg({'tmean': 'sum', 'ppt': 'sum'}).reset_index()
    tdat = tdat.rename(columns={'adj_year': 'year'})
    return tdat


def get_snotel(gdf, month, min_year, max_year):
    # Get snotel data
    wsdlurl = 'https://hydroportal.cuahsi.org/Snotel/cuahsi_1_1.asmx?WSDL'
    sites = ulmo.cuahsi.wof.get_sites(wsdlurl)
    sites_df = pd.DataFrame.from_dict(sites, orient='index').dropna()
    in_sites = filter_snotel(gdf, sites_df)

    in_sites = [x.replace("_", ":") for x in in_sites]

    snotel_dat = [proc_snotel(x) for x in in_sites]

    # Initialize the merged dataframe with the first dataframe in the list
    merged_snotel_df = snotel_dat[0]

    # Iterate over the remaining dataframes and merge them one by one
    for df in snotel_dat[1:]:
        merged_snotel_df = pd.merge(merged_snotel_df, df, on=['date', 'year', 'month'], how='outer')

    snotel_dat = merged_snotel_df[(merged_snotel_df['year'] >= min_year) & (merged_snotel_df['year'] <= max_year)]

    snotel_dat = snotel_dat[snotel_dat['month'] == month]

    snotel_dat = snotel_dat.drop(columns=['date', 'month'])

    snotel_dat = snotel_dat.sort_values('year').reset_index(drop=True)

    # Keeping only columns with no NA values
    snotel_dat = snotel_dat.dropna(axis=1, how='any')

    snotel_dat = snotel_dat.dropna()

    return snotel_dat


def get_aso(watershed):
    aso_dat = pd.read_csv(f"data/{watershed}/processed/aso_basin_data.csv")

    aso_dat = aso_dat.assign(year = pd.to_datetime(aso_dat['date']).dt.strftime("%Y"),
        month = pd.to_datetime(aso_dat['date']).dt.strftime("%m"))

    adat = aso_dat.groupby(['year', 'month']).agg({'SWE': 'mean'}).reset_index()

    adat = adat.assign(month = adat['month'].astype(int),
        year = adat['year'].astype(int))

    adat = adat.groupby(['year']).agg({'SWE': 'mean'}).reset_index()


    return adat


def setup_dirs(watershed):
    # Setup M4 directories
    os.system(f"mkdir ~/Projects/M4/examples/{watershed}_baseline/")
    os.system(f"mkdir ~/Projects/M4/examples/{watershed}_aso_swe/")
    os.system(f"mkdir ~/Projects/M4/examples/{watershed}_aso_swe_total/")
    os.system(f"mkdir ~/Projects/M4/examples/{watershed}_baseline_aso_swe/")
    os.system(f"mkdir ~/Projects/M4/examples/{watershed}_baseline_aso_swe_temp_precip/")

    # Copy skeleton directories
    os.system(f"cp -R ~/Projects/M4/examples/baseline/* ~/Projects/M4/examples/{watershed}_baseline/.")
    os.system(f"cp -R ~/Projects/M4/examples/baseline/* ~/Projects/M4/examples/{watershed}_aso_swe/.")
    os.system(f"cp -R ~/Projects/M4/examples/baseline/* ~/Projects/M4/examples/{watershed}_aso_swe_total/.")
    os.system(f"cp -R ~/Projects/M4/examples/baseline/* ~/Projects/M4/examples/{watershed}_baseline_aso_swe/.")
    os.system(f"cp -R ~/Projects/M4/examples/baseline/* ~/Projects/M4/examples/{watershed}_baseline_aso_swe_temp_precip/.")


def merge_save(watershed, snotel_dat, streamGage_dat, prism_dat, num_bins=10):
    print("Starting merge_save function")

    # Merge stream gauge data with SNOTEL data on 'year' column and drop missing values
    mdat = streamGage_dat.merge(snotel_dat, on=['year'], how='left')
    mdat = mdat.dropna().reset_index(drop=True)
    print("Merged streamGage data and snotel data")

    # Save the merged data as baseline model data
    mdat.to_csv(f"~/Projects/M4/examples/{watershed}_baseline/model_data_baseline.txt", sep='\t', index=False)
    mdat.to_csv(f"~/Projects/M4/examples/{watershed}_baseline/MMPEInputData_ModelBuildingMode.txt", sep='\t', index=False)
    print("Saved baseline model data")

    # Load ASO ASW prediction data
    aso_pred = glob.glob(f"data/{watershed}/predictions/NN*.parquet")[0]
    adat = pd.read_parquet(aso_pred)
    print("Loaded ASO ASW prediction data")

    # Determine minimum and maximum elevation from ASO ASW data
    min_elevation = adat['elevation'].min()
    max_elevation = adat['elevation'].max()
    print("Determined min and max elevation")

    # Create bins for elevation
    bins = np.linspace(min_elevation, max_elevation, num_bins)
    adat['elevation_bin'] = pd.cut(adat['elevation'], bins, labels=False, include_lowest=True)
    print("Created elevation bins")

    # Aggregate ASO ASW data by year and elevation bin
    adat = adat.groupby(['year', 'elevation_bin']).agg({'swe_pred': np.sum}).reset_index()
    adat = adat.sort_values('year')
    adat = adat.pivot(index='year', columns='elevation_bin', values='swe_pred').add_prefix('elevbin_').reset_index()
    print("Aggregated and pivoted ASO ASW data")

    # Merge stream gauge data with ASO ASW data and save
    mdat = streamGage_dat.merge(adat, on=['year'], how='left')
    mdat = mdat.dropna().reset_index(drop=True)
    mdat.to_csv(f"~/Projects/M4/examples/{watershed}_aso_swe/model_data_aso_swe.txt", sep='\t', index=False)
    mdat.to_csv(f"~/Projects/M4/examples/{watershed}_aso_swe/MMPEInputData_ModelBuildingMode.txt", sep='\t', index=False)
    print("Merged and saved stream gauge data with ASO ASW data")

    # Merge for baseline and ASO SWE
    mdat = streamGage_dat.merge(snotel_dat, on=['year'], how='left')
    mdat = mdat.merge(adat, on=['year'], how='left')
    mdat = mdat.dropna().reset_index(drop=True)
    mdat.to_csv(f"~/Projects/M4/examples/{watershed}_baseline_aso_swe/model_data_aso_swe.txt", sep='\t', index=False)
    mdat.to_csv(f"~/Projects/M4/examples/{watershed}_baseline_aso_swe/MMPEInputData_ModelBuildingMode.txt", sep='\t', index=False)
    print("Merged and saved baseline and ASO SWE data")

    # Merge for baseline, ASO SWE, and PRISM temperature and precipitation data
    mdat = streamGage_dat.merge(snotel_dat, on=['year'], how='left')
    mdat = mdat.merge(prism_dat, on=['year'], how='left')
    mdat = mdat.merge(adat, on=['year'], how='left')
    mdat = mdat.dropna().reset_index(drop=True)
    mdat.to_csv(f"~/Projects/M4/examples/{watershed}_baseline_aso_swe_temp_precip/model_data_aso_swe.txt", sep='\t', index=False)
    mdat.to_csv(f"~/Projects/M4/examples/{watershed}_baseline_aso_swe_temp_precip/MMPEInputData_ModelBuildingMode.txt", sep='\t', index=False)
    print("Merged and saved baseline, ASO SWE, temp, and precip data")

    print("merge_save function completed")



if __name__ == "__main__":

    # Hetchy flow data
    # sdat = pd.read_csv("data/CDEC/hh_Q_Monthly.csv")
    # sdat.columns = ['year-month', 'kaf']

    # sdat['month'] = [int(x.split(" ")[1]) for x in sdat['year-month']]
    # sdat['year'] = [int(x.split(" ")[0]) for x in sdat['year-month']]

    # sdat = sdat[(sdat['month'] >= 4) & (sdat['month'] <= 7)]
    # sdat = sdat.groupby('year').agg({'kaf': np.sum}).reset_index()

    # sdat['kaf'] = sdat['kaf']/1000
    # sdat



    # Setup directories for processing data for region
    dir_name = "Tuolumne_Watershed"
    watershed = "Tuolumne_Watershed"

    min_year = 1981
    max_year = 2022
    
    # ---------------------------------------------
    # Blue Dillon
    watershed = "Blue_Dillon_Watershed"
    shapefile_loc = glob.glob(f"data/{watershed}/shapefiles/*.shp")[0]

    gdf = gpd.read_file(shapefile_loc)
    gdf = gdf.to_crs(epsg=4326)
    
    setup_dirs(watershed)

    snotel_dat = get_snotel(gdf, 4, min_year, max_year)
    streamGage_dat = get_flowrate("09050700:CO:USGS", "1981-01-01", "2023-09-30")
    prism_dat = get_prism(watershed, min_year, max_year)
    merge_save(watershed, snotel_dat, streamGage_dat, prism_dat, num_bins=10)


    path = glob.glob(f"data/{watershed}/predictions/*")[0]
    len(pd.read_parquet(path))

    len(pd.read_csv(f"data/{watershed}/processed/aso_basin_data.csv"))

    # ---------------------------------------------
    # Dolores
    watershed = "Dolores_Watershed"
    min_year = 1987
    max_year = 2022

    shapefile_loc = glob.glob(f"data/{watershed}/shapefiles/*.shp")[0]

    gdf = gpd.read_file(shapefile_loc)

    setup_dirs(watershed)

    snotel_dat = get_snotel(gdf, 4, min_year, max_year)
    streamGage_dat = get_flowrate("09166500:CO:USGS", "1981-01-01", "2023-09-30")
    prism_dat = get_prism(watershed, min_year, max_year)
    merge_save(watershed, snotel_dat, streamGage_dat, prism_dat, num_bins=4)

    # ---------------------------------------------
    # Conejos
    watershed = "Conejos_Watershed"
    shapefile_loc = glob.glob(f"data/{watershed}/shapefiles/*.shp")[0]

    min_year = 1981
    max_year = 2022

    gdf = gpd.read_file(shapefile_loc)
    gdf = gdf.to_crs(epsg=4326)

    setup_dirs(watershed)

    # No Snotel stations
    # snotel_dat = get_snotel(gdf)

    # Lily Pond (not in grid, but in Conejos county)
    # is_point_in_geodf(gdf, 37.38, -106.55)

    # Cumbres Treselt (not in grid, but in Conejos county)
    # is_point_in_geodf(gdf, 37.02, -106.45)

    snow1 = proc_snotel("580:CO:SNTL")
    snow2 = proc_snotel("431:CO:SNTL")

    snotel_dat = snow1.merge(snow2, on=['date', 'year', 'month'], how='left')
    snotel_dat = snotel_dat.drop(columns=['date', 'month'])
    snotel_dat = snotel_dat.sort_values('year')
 
    streamGage_dat = get_flowrate("08246500:CO:USGS", "1981-01-01", "2023-09-30")
    prism_dat = get_prism(watershed, min_year, max_year)
    merge_save(watershed, snotel_dat, streamGage_dat, prism_dat, num_bins=8)




    # ------------------------------------------------
    # For conda environment reqs
    # conda create --name M4 r=3.6
    # conda activate M4

    # sudo apt-get install libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev

    # conda install -c conda-forge r-forecast r-qrnn r-e1071 r-akima r-genalg r-doParallel r-foreach 
    #                              r-quantreg r-quantregGrowth r-matrixStats r-randomForest r-nloptr

    # Rscript -e "install.packages('monmlp', repos='http://cran.rstudio.com/')"

    # rm Rplot*
    # Rscript MMPE-Main_MkII.R






