import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import os

# import ulmo   # snotel data

from libs.asolibs import *
from libs.gagelibs import StreamGauge
# from libs.prismlibs import *


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



# Setup directories for processing data for region
dir_name = "Tuolumne_Watershed"
os.system(f"mkdir data/{dir_name}")


# -----------------------------------------------------------
# Load shapefile
shape_loc = "data/shapefiles/tuolumne_watershed/Tuolumne_Watershed.shp"
gage_id = "11290000"

gdf = gpd.read_file(shape_loc)
gdf = gdf.to_crs(epsg=4326)




# -----------------------------------------------------------
# Get streamgage data
sdat = StreamGauge(site = gage_id, start_date = "1950-01-01", end_date = "2023-01-02").proc_json()

# Filter out discharge
sdat = sdat[sdat['var_desc'] == "Discharge, cubic feet per second"]
sdat = sdat.assign(month = pd.to_datetime(sdat['date']).dt.strftime("%m"),
                   year = pd.to_datetime(sdat['date']).dt.strftime("%Y"))

sdat = sdat.assign(month = sdat['month'].astype(int),
                   year = sdat['year'].astype(int))

# March-July, or April-July
sdat = sdat[(sdat['month'] >= 3) & (sdat['month'] <= 7)]

# sdat = sdat.assign(kaf = sdat['value'].astype(float) * 0.00002295684)
sdat = sdat.assign(kaf = sdat['value'].astype(float) * (1.98347/1000))
sdat = sdat.groupby('year').agg({'kaf': 'sum'}).reset_index()

sdat.to_csv(f"data/{dir_name}/{dir_name}_streamgage. csv", index=False)

min_year = sdat['year'].min()
max_year = sdat['year'].max()





# -----------------------------------------------------------
# Get prism data
# proc_daily(gdf, "Tuolumne_Watershed")
prism_dat = pd.read_csv("data/Tuolumne_Watershed_PRISM_daily_1981-2022.csv")
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
dat2 = pivoted_df[pivoted_df['month'] <= 7]
tdat = pd.concat([dat1, dat2])

tdat['adj_year'] = np.where(tdat['month'] >= 10, tdat['year'] + 1, tdat['year'])
tdat['tmean'] = (tdat['tmax'] + tdat['tmin']) / 2

tdat = tdat.groupby(['adj_year']).agg({'tmean': 'sum', 'ppt': 'sum'}).reset_index()
tdat.to_csv(f"data/{dir_name}/{dir_name}_PRISM_data.csv", index = False)




# -----------------------------------------------------------
# Get snotel data
wsdlurl = 'https://hydroportal.cuahsi.org/Snotel/cuahsi_1_1.asmx?WSDL'
sites = ulmo.cuahsi.wof.get_sites(wsdlurl)
sites_df = pd.DataFrame.from_dict(sites, orient='index').dropna()
in_sites = filter_snotel(gdf, sites_df)
new_sites_df = sites_df[sites_df['code'].isin(in_sites)]
new_sites_df.to_csv(f"data/{dir_name}/{dir_name}_SNOTEL_data.csv", index = False)




# -----------------------------------------------------------
# Get ASO data
aso_dat = proc_ASO_SWE_shp(gdf)
aso_dat = aso_dat.dropna()
aso_dat = aso_dat[aso_dat['SWE'] >= 0]

aso_dat = aso_dat.assign(year = pd.to_datetime(aso_dat['date']).dt.strftime("%Y"),
    month = pd.to_datetime(aso_dat['date']).dt.strftime("%m"))

aso_dat.to_csv(f"data/{dir_name}/{dir_name}_data.csv", index=False)

adat = aso_dat.groupby(['year', 'month']).agg({'SWE': 'mean'}).reset_index()

adat = adat.assign(month = adat['month'].astype(int),
    year = adat['year'].astype(int))

# adat = adat[adat['month'] <= 4]

adat = adat.groupby(['year']).agg({'SWE': 'mean'}).reset_index()

#     year month           SWE
# 0   2013    04  2.745523e+05
# 1   2013    05  1.166111e+05
# 2   2013    06  3.769991e+04
# 3   2016    03  3.027516e+05
# 4   2016    04  1.358669e+06
# 5   2016    05  4.026820e+05
# 6   2016    07  6.100502e+03
# 7   2017    01  8.939879e+05
# 8   2017    07  8.967773e+04
# 9   2017    08  9.619107e+03
# 10  2018    04  2.473785e+05
# 11  2018    05  7.054485e+04
# 12  2019    03  6.978560e+05
# 13  2019    04  6.991465e+05
# 14  2019    05  5.159346e+05
# 15  2019    06  2.695899e+05
# 16  2019    07  6.605940e+04
# 17  2020    04  3.583343e+05
# 18  2020    05  3.654243e+05
# 19  2021    02  2.560861e+05
# 20  2021    04  1.125345e+05
# 21  2022    02  4.562076e+05
# 22  2022    04  2.764274e+05
# 23  2022    05  6.779469e+04
# 24  2023    01  6.095430e+05
# 25  2023    03  1.878401e+06
# 26  2023    04  1.029180e+06






# aso_dat = aso_dat.groupby('year').agg({'SWE': 'sum'}).reset_index()

# aso_dat.to_csv(f"data/{dir_name}/{dir_name}_ASO_data.csv", index = False)



# Bind all together
mdat = sdat.merge(tdat, left_on=['year'], right_on=['adj_year'], how='left').drop(columns='adj_year')
mdat = adat.merge(mdat, left_on=['year'], right_on=['year'], how='left').dropna().reset_index(drop=True)
mdat = mdat[['year', 'kaf', 'tmean', 'ppt', 'SWE']]

mdat.to_csv('data/Tuolumne_Watershed/tuolumne_model_data.txt', sep='\t', index=False)


