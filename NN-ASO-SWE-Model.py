import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.metrics import r2_score, mean_squared_error



def adapt_learning_rate(epoch):
    return 0.001 * epoch

# -----------------------------------
# Build model data

mdat = pd.read_parquet("data/Tuolumne_Watershed/model_data_elevation_prism_sinceSep_nlcd.parquet")
mdat = mdat.dropna()

mdat = mdat.assign(year = pd.to_datetime(mdat['date']).dt.strftime("%Y"),
    month = pd.to_datetime(mdat['date']).dt.strftime("%m"))

mdat.head()

mdat['lat_x_lon'] = mdat['lat_x'] * mdat['lon_x']

mdat.columns

X = mdat[['snow', 'tmean', 'tmax', 'tmin', 'ppt', 'month', 
          'lat', 'lon', 'lat_x_lon', 'elevation', 'landcover']]

# X = mdat[['snow', 'month', 'lat_x', 'lon_x']]
y = mdat[['SWE']]

months = pd.get_dummies(X['month'])
landcover = pd.get_dummies(X['landcover'])

X = pd.concat([X, months, landcover], axis=1)
X = X.drop(columns=['month', 'landcover'])

# Assume X, y are your features and target values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -----------------------------------
# Build Model

# Set early stop
es = EarlyStopping(monitor='val_loss', patience=3, verbose=0,
    restore_best_weights=True)

# Set adaptive learning rate
my_lr_scheduler = LearningRateScheduler(adapt_learning_rate)


mod = Sequential()
# mod.add(Dense(60, activation='relu'))
# mod.add(Dense(30, activation='relu'))
mod.add(Dense(15, activation='relu'))
mod.add(Dense(10, activation='relu'))
mod.add(Dense(5, activation='relu'))
mod.add(Dense(1, activation='linear'))
mod.compile(optimizer='Adam', loss='mean_squared_error') 
mod.fit(X_train_scaled, y_train.values, validation_split=0.2,
    shuffle=True, epochs=1000,  batch_size=128,
    callbacks=[es, my_lr_scheduler]
    )

# Use the model for prediction
y_train_pred = mod.predict(X_train_scaled)
y_test_pred = mod.predict(X_test_scaled)

r2_score(y_train, y_train_pred)
r2_score(y_test, y_test_pred)

np.sqrt(mean_squared_error(y_train, y_train_pred))
np.sqrt(mean_squared_error(y_test, y_test_pred))

# Build output data frame for figures
outdat = pd.DataFrame({'lat': X_train['lat_x'],
    'lon': X_train['lon_x'],
    'y_pred': y_train_pred.ravel(),
    'y_true': y_train['SWE']})
outdat['residuals'] = outdat['y_pred'] - outdat['y_true']

outdat = outdat.assign(lat_lon = outdat['lat'].astype(str) + "_" + outdat['lon'].astype(str))
outdat = outdat.groupby('lat_lon').agg({'lat': np.mean, 'lon': np.mean, 'residuals': np.mean}).reset_index()
 
outdat.to_csv("data/test.csv", index=False)

outdat = pd.DataFrame({'lat': X_test['lat_x'],
    'lon': X_test['lon_x'],
    'y_pred': y_test_pred.ravel(),
    'y_true': y_test['SWE']})

outdat['residuals'] = outdat['y_pred'] - outdat['y_true']

outdat = outdat.assign(lat_lon = outdat['lat'].astype(str) + "_" + outdat['lon'].astype(str))
outdat = outdat.groupby('lat_lon').agg({'lat': np.mean, 'lon': np.mean, 'residuals': np.mean})

outdat.to_csv("data/test2.csv", index=False)



# -----------------------------------
# Cross-validate by year

mdat = mdat.assign(year = pd.to_datetime(mdat['date']).dt.year)

X = mdat[['year', 'snow', 'tmean', 'tmax', 'tmin', 'ppt', 'month', 
          'lat', 'lon', 'lat_x_lon', 'elevation', 'landcover']]

# X = mdat[['snow', 'month', 'lat_x', 'lon_x']]
y = mdat[['SWE']]

months = pd.get_dummies(X['month'])
landcover = pd.get_dummies(X['landcover'])

X = pd.concat([X, months, landcover], axis=1)
X = X.drop(columns=['month', 'landcover'])

# Assuming your data has a 'year' column to split by
unique_years = X['year'].unique()

results = []
for year in unique_years:
    # Split data by year
    X_train = X[X['year'] != year]
    y_train = y[X['year'] != year]
    X_test = X[X['year'] == year]
    y_test = y[X['year'] == year]
    
    X_train = X_train.drop(columns=['year'])
    X_test = X_test.drop(columns=['year'])

    # Scale the features
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    mod = Sequential()
    mod.add(Dense(15, activation='relu'))
    mod.add(Dense(10, activation='relu'))
    mod.add(Dense(5, activation='relu'))
    mod.add(Dense(1, activation='linear'))
    mod.compile(optimizer='Adam', loss='mean_squared_error')
    
    mod.fit(X_train_scaled, y_train.values, validation_split=0.2,
        shuffle=True, epochs=1000,  batch_size=128,
        callbacks=[es, my_lr_scheduler]
        )
        
    # Use the model for prediction
    y_train_pred = mod.predict(X_train_scaled)
    y_test_pred = mod.predict(X_test_scaled)
    
    # Compute and store metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    results.append({
        'year': year,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse
    })






