import pandas as pd
import xarray as xr
import numpy as np
from scipy.interpolate import griddata

df = pd.read_csv("data/Tuolumne_Watershed/aso_elevation.csv")
df = df[['lat', 'lon', 'elevation']]

# Pivot the DataFrame to create a 2D array with NaNs where data is missing
elevation_2d = df.pivot(index='lat', columns='lon', values='elevation')

# Define a grid of latitude and longitude values
lat_unique = np.linspace(df['lat'].min(), df['lat'].max(), num_lats)
lon_unique = np.linspace(df['lon'].min(), df['lon'].max(), num_lons)

lat_grid, lon_grid = np.meshgrid(lat_unique, lon_unique)

# Interpolate the elevation data onto the regular grid
elevation_interpolated = griddata(
    points=df[['lat', 'lon']].values,
    values=df['elevation'].values,
    xi=(lat_grid, lon_grid),
    method='cubic'
)

# Get the number of unique latitudes and lonitudes
num_lats, num_lons = elevation_2d.shape

# Create an xarray DataArray from your pivoted data
lat = elevation_2d.index.values
lon = elevation_2d.columns.values

elevation = xr.DataArray(elevation_2d.values, dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon})

# Create a new xarray Dataset
ds = xr.Dataset({'elevation': elevation})

# Compute the gradient in the x and y directions
gradient_x, gradient_y = np.gradient(ds.elevation.values, axis=(0, 1))

# Compute the grade and aspect
grade = np.sqrt(gradient_x**2 + gradient_y**2)
aspect = np.arctan2(-gradient_y, gradient_x) * 180.0 / np.pi

# Save the computed grade and aspect as new DataArrays in your xarray Dataset
ds['grade'] = xr.DataArray(grade, dims=('lat', 'lon'), coords={'lat': ds.lat.values, 'lon': ds.lon.values})
ds['aspect'] = xr.DataArray(aspect, dims=('lat', 'lon'), coords={'lat': ds.lat.values, 'lon': ds.lon.values})

print(ds)

outdat = ds.to_dataframe().reset_index()
outdat = outdat.dropna()

len(df)
len(outdat)
