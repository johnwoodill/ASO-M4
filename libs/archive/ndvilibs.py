def get_ndvi_filelist(year):

    url = f"https://www.ncei.noaa.gov/data/land-normalized-difference-vegetation-index/access/{year}/"

    # Fetch the content of the webpage
    response = requests.get(url)

    # Parse the webpage content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all the links in the webpage
    links = soup.find_all('a')

    # Print the URLs of the files
    lst_ = []
    for link in links:
        href = link.get('href')
        if href:  # Check if the href attribute is not None
            file_url = url + href
            lst_.append(file_url)
    lst_ = lst_[5:]

    # Define start and end dates as strings in the format YYYYMMDD
    start_date = f"{year}0101"
    # start_date = f"{year}0601"
    end_date = f"{year}0731"

    # Filter the list of files
    filtered_urls = []
    for file_url in lst_:
        # Extract the date string from the file name (assuming it's always in the same position)
        date_string = file_url.split('_')[-2]  # Split by underscore and get the second to last part
        
        # Check if the date_string is within the desired range
        if start_date <= date_string <= end_date:
            filtered_urls.append(file_url)

    return filtered_urls


def proc_ndvi(year, gdf):
    
    files = get_ndvi_filelist(year)
    max_retry = 5  # Define max retry attempts for downloading a file.

    dat_ = []
    for url in files:
        success = False
        attempts = 0
        while not success and attempts < max_retry:
            try:
                print(f"Attempt {attempts+1}: Downloading {url}")
                # Download the netCDF file
                filename = url.split("/")[-1]
                os.system(f"wget {url} -O {filename}")

                # Try to open the netCDF file using xarray
                ds = xr.open_dataset(filename)
                success = True
            except Exception as e:
                attempts += 1
                print(f"Failed to download or open file {url}, retrying ({attempts}/{max_retry})")
                if attempts < max_retry:
                    time.sleep(60)  # Sleep for 1 minute before retrying

        if success:
            try:
                ds = xr.open_dataset(filename)
                ds = ds['NDVI']
                ds = ds.rio.write_crs("EPSG:4326")

                # Proceed with processing
                dat = ds.rio.clip(gdf.geometry, all_touched=True, drop=True, invert=False, from_disk=True)
                dat = dat.to_series().reset_index()
                outdat = dat.dropna()
                dat_.append(outdat)
                
                # Optionally delete the file after processing
                os.system(f"rm {filename}")

            except Exception as e:
                print(f"Error processing file {url}: {e}")

    if dat_:
        # Save the results to a CSV file if there's any data
        rdat = pd.concat(dat_)
        rdat.to_csv(f"data/{basin_name}/NDVI/{year}.csv", index=False)
    else:
        print(f"No data available for year {year}")


def get_ndvi(lat, lon, start_date, end_date):
    """
    Fetch NDVI data for a given latitude/longitude coordinate and date range.
    
    Args:
        lat (float): The latitude of the coordinate.
        lon (float): The longitude of the coordinate.
        start_date (str): The start date for fetching data in 'YYYY-MM-DD' format.
        end_date (str): The end date for fetching data in 'YYYY-MM-DD' format.
    
    Returns:
        list: A list of tuples containing date and NDVI values for each day within the date range.
    """
    def apply_rolling_mean(ndvi_values, window_size=3):
        """
        Apply rolling mean to the NDVI values.
        
        Args:
            ndvi_values (list): List of tuples containing date and NDVI values.
            window_size (int): Size of the rolling window.
        
        Returns:
            list: A list of tuples containing date and rolling mean NDVI values.
        """
        rolling_ndvi = []
        values = [val[1] for val in ndvi_values]
        for i in range(len(ndvi_values)):
            if i < window_size // 2 or i >= len(ndvi_values) - window_size // 2:
                rolling_ndvi.append((ndvi_values[i][0], "NA"))
            else:
                mean = sum(values[i - window_size // 2: i + window_size // 2 + 1]) / window_size
                rolling_ndvi.append((ndvi_values[i][0], mean))
        return rolling_ndvi

    # Define a point geometry from the coordinates
    point = ee.Geometry.Point(lon, lat)
    
    # Use the MODIS NDVI dataset
    dataset = ee.ImageCollection('MODIS/006/MOD13A2').filterDate(start_date, end_date)
    
    # Initialize an empty list to store the results
    ndvi_values = []
    
    # Iterate over each day within the date range
    date = start_date
    while date <= end_date:
        # Calculate the end date as the next day
        next_date = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        
        # Filter the dataset for the current day
        daily_dataset = dataset.filter(ee.Filter.date(date, next_date))
        
        # Calculate the average NDVI for the current day
        ndvi = daily_dataset.select('NDVI').mean()
        
        try:
            # Sample the NDVI at the point location and get the result
            result = ndvi.sample(region=point, scale=500, projection='EPSG:4326').first().get('NDVI').getInfo()
            ndvi_values.append((date, result * 0.0001))
        except Exception as e:
            print(f"Failed to retrieve NDVI for date {date}: {str(e)}")
            ndvi_values.append((date, "NA"))
        
        # Increment the date to the next day
        date = next_date
    
    # Apply rolling mean through the NDVI values
    ndvi_values = apply_rolling_mean(ndvi_values)
    
    return ndvi_values


def process_ndvi_data(dataframe):
    """
    Process a DataFrame to fetch NDVI data for each row.
    
    Args:
        dataframe (pd.DataFrame): DataFrame containing 'lat', 'lon', and 'date' columns.
    
    Returns:
        pd.DataFrame: DataFrame with an additional 'ndvi' column.
    """
    
    # Function to apply to each row of the DataFrame
    def fetch_ndvi(row):
        return get_ndvi(row[2], row[3], row[1])
    
    # Apply the fetch_ndvi function to each row and add a progress bar
    ndvi_values = [fetch_ndvi(row) for row in tqdm(dataframe.itertuples(), total=len(dataframe), desc="Fetching NDVI data")]
    
    # Add the NDVI values as a new column to the DataFrame
    dataframe['ndvi'] = ndvi_values
    
    return dataframe
