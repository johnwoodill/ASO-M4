def get_elevation(lat, lon, num_retries=10):
    url = 'https://epqs.nationalmap.gov/v1/json?'
    params = {
        'output': 'json',
        'x': lon,
        'y': lat,
        'units': 'Meters'
    }
    for i in range(num_retries):
        try:
            result = requests.get((url + urllib.parse.urlencode(params)))
            return result.json()['value']
        except Exception as e:
            print(f"Failed to retrieve elevation for coordinates ({lat}, {lon}): {str(e)}. Retry attempt: {i+1}")
            time.sleep(10)  # wait for 1 second before retrying
    print(f"Failed to retrieve elevation after {num_retries} attempts")
    return "NA"


def elevation_function(df, lat_column, lon_column, main_directory, max_workers=8):
    """Query service using lat, lon. Save individual files for each coordinate."""

    def save_elevation(lat, lon):
        file_name =  f"{lat:.4f}_{lon:.4f}.csv"
        file_path = os.path.join(main_directory, file_name)

        # Check if the file already exists
        if os.path.isfile(file_path):
            # Uncomment the next line if you want to print a message for existing files
            # print(f"File '{file_name}' already exists. Skipping.")
            return

        elevation = get_elevation(lat, lon)

        if elevation == "NA":
            return
        
        # Save the elevation as a separate file
        with open(file_path, 'w') as file:
            file.write("Latitude,Longitude,Elevation\n")
            file.write(f"{lat},{lon},{elevation}\n")

    def fetch_elevation(row):
        lat = row[1][lat_column]
        lon = row[1][lon_column]
    
        save_elevation(lat, lon)

    # Using ThreadPoolExecutor to parallelize the requests
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Wrap the executor.map with tqdm for a progress bar
        _ = list(tqdm(executor.map(fetch_elevation, df.iterrows()), total=len(df), desc="Fetching elevation data"))



def proc_elevation(gdf, basin_name):
    
    print(f"Getting ASO data for {basin_name}")
    
    # Get unique lat/lon
    lat_lon_dat = pd.read_csv(f"data/{basin_name}/processed/aso_basin_data.csv")
    lat_lon_dat = lat_lon_dat.drop_duplicates(subset=['lat_lon'])

    # lat_lon_dat = lat_lon_dat.iloc[0:20, :]

    # print("Checking for already processed files")
    elev_files = glob.glob(f"data/{basin_name}/elevation/*.csv")
    lat_lon_files = [x.split("/")[-1].replace(".csv", "") for x in elev_files] 

    lat_lon_dat = lat_lon_dat[~lat_lon_dat['lat_lon'].isin(lat_lon_files)]

    if len(lat_lon_dat) > 0:
        print(f"Processing ASO elevation data")
        elevation_function(lat_lon_dat, 'lat', 'lon', f"data/{basin_name}/elevation/", 50)
    else:
        print("No elevations to process. Binding data")

    print(f"Binding elevation data")
    elev_files = glob.glob(f"data/{basin_name}/elevation/*.csv")

    len(elev_files)
    len(lat_lon_dat['lat_lon'].unique())    

    df_ = []
    for file_ in tqdm(elev_files):
        df = pd.read_csv(file_)
        df_.append(df)

    eldat = pd.concat(df_)
    eldat.to_parquet(f"data/{basin_name}/processed/aso_elevation.parquet", compression=None, index=False)
    print(f"Saved: data/{basin_name}/processed/aso_elevation.parquet")
