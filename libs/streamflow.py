import geopandas as gpd 
import pandas as pd 
import numpy as np
from urllib import request
import json
import fiona

def get_json(url):
    with request.urlopen(url) as url_:
        djson = json.loads(url_.read().decode())
    return djson


def find_downstream_route(coords):
    """ 
    Retrun the json of the flowlines for the downstream route using the USGS API river;
    Arg:
        coords: lat/lon coordinates
    """
    url = f"https://labs.waterdata.usgs.gov/api/nldi/linked-data/hydrolocation?coords=POINT%28{coords[1]}%20{coords[0]}%29"
    djson = get_json(url)

    # Get navigation location urls
    navurl = djson['features'][0]['properties']['navigation']
    navjson = get_json(navurl)

    # Get downstream url
    ds_main = navjson['downstreamMain']
    
    # Get downstream data
    downstream_main = get_json(ds_main)
    ds_flow = downstream_main[0]['features']
    with_distance = ds_flow + '?distance=5500'
    flowlines = get_json(with_distance)
    gdf = gpd.GeoDataFrame.from_features(flowlines)
    print(f"Num of features = {len(flowlines['features'])}")
    return gdf



# dat = gpd.read_file("data/WBD/Shape/WBDHU12.shp")

# dat = dat[dat['name'].str.contains("Tuol")]

# dat.to_file("test.shp")