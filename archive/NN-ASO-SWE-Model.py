import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.metrics import r2_score, mean_squared_error
from keras.regularizers import l2



def adapt_learning_rate(epoch):
    return 0.001 * epoch

# -----------------------------------
# Build model data

mdat = pd.read_parquet("data/Tuolumne_Watershed/model_data_elevation_prism_sinceSep_nlcd_V2.parquet")

mdat.dtypes

mdat = mdat.assign(year = pd.to_datetime(mdat['date']).dt.strftime("%Y"),
    month = pd.to_datetime(mdat['date']).dt.strftime("%m"),
    doy = pd.to_datetime(mdat['date']).dt.strftime("%j"))

mdat.head()

# mdat.columns

# X = mdat[['snow', 'tmean', 'ppt', 'month', 'doy',
#           'lat', 'lon', 'lat_x_lon', 'elevation', 'landcover']]

# # X = mdat[['snow', 'month', 'lat_x', 'lon_x']]
# y = mdat[['SWE']]

# months = pd.get_dummies(X['month'])
# landcover = pd.get_dummies(X['landcover'])

# X = pd.concat([X, months, landcover], axis=1)
# X = X.drop(columns=['month', 'landcover'])

# # Assume X, y are your features and target values
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Scale the features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)


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

# Set early stop
es = EarlyStopping(monitor='val_loss', patience=3, verbose=0,
    restore_best_weights=True)

# Set adaptive learning rate
my_lr_scheduler = LearningRateScheduler(adapt_learning_rate)

# mdat = mdat.assign(year = pd.to_datetime(mdat['date']).dt.year)

rm_dates = ["2019-03-29", "2019-06-04", "2019-07-03", "2019-07-16", "2018-04-25"]

moddat = mdat[~mdat['date'].isin(rm_dates)]

# moddat = moddat[moddat['year'] <= 2020]

# moddat = moddat[moddat['month'] <= "3"]

# X = moddat[['year', 'snow', 'tmean', 'tmax', 'tmin', 'ppt', 'month', 
#           'lat', 'lon', 'lat_x_lon', 'elevation', 'landcover']]

# X = moddat[['year', 'snow', 'tmean', 'ppt', 'month', 'lat', 'lon',
#           'elevation', 'landcover']]

X = moddat[['year', 'snow', 'tmean', 'ppt', 'month', 'doy', 
          'lat', 'lon', 'lat_x_lon', 'elevation', 'landcover']]

# X = moddat[['snow', 'month', 'lat_x', 'lon_x']]
y = moddat[['SWE']]

months = pd.get_dummies(X['month'])
landcover = pd.get_dummies(X['landcover'])

X = pd.concat([X, months, landcover], axis=1)
X = X.drop(columns=['month', 'landcover'])

# Assuming your data has a 'year' column to split by
unique_years = X['year'].unique()

results = []

year = 2013
for year in unique_years:
    print(f"Processing: {year}")

    # Split data by year
    X_train = X[X['year'] != year]
    y_train = y[X['year'] != year]

    X_test = X[X['year'] == year]
    y_test = y[X['year'] == year]
    
    X_train = X_train.drop(columns=['year'])
    X_test = X_test.drop(columns=['year'])

    print("Scaling data")
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # mod = Sequential()
    # mod.add(Dense(15, activation='relu'))
    # mod.add(Dense(10, activation='relu'))
    # mod.add(Dense(5, activation='relu'))
    # mod.add(Dense(1, activation='linear'))
    # mod.compile(optimizer='Adam', loss='mean_squared_error')


    # mod = Sequential()
    # mod.add(Dense(15, activation='relu', kernel_regularizer=l2(0.01)))
    # mod.add(Dense(15, activation='relu')
    # mod.add(Dropout(0.80))
    # mod.add(Dense(10, activation='relu', kernel_regularizer=l2(0.01)))
    # mod.add(Dense(10, activation='relu')
    # mod.add(Dropout(0.80))
    # mod.add(Dense(5, activation='relu'))
    # mod.add(Dense(1, activation='linear'))
    # mod.compile(optimizer='Adam', loss='mean_squared_error')
    # mod.fit(X_train_scaled, y_train.values, validation_split=0.2,
    #     shuffle=True, epochs=1000,  batch_size=1024*2,
    #     callbacks=[es, my_lr_scheduler]
    #     )

    mod = Sequential()
    mod.add(Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)))
    mod.add(BatchNormalization())
    # mod.add(Dropout(0.6))

    mod.add(Dense(32, activation='relu', kernel_regularizer=l2(0.005)))
    mod.add(BatchNormalization())
    # mod.add(Dropout(0.5))

    mod.add(Dense(16, activation='relu', kernel_regularizer=l2(0.005)))
    mod.add(BatchNormalization())
    mod.add(Dropout(0.4))

    mod.add(Dense(8, activation='relu'))
    mod.add(BatchNormalization())
    mod.add(Dropout(0.3))

    mod.add(Dense(1, activation='linear'))

    mod.compile(optimizer='Adam', loss='mean_squared_error')

    mod.fit(X_train_scaled, y_train.values, validation_split=0.2, 
            shuffle=True, epochs=1000, batch_size=1024*2, 
            callbacks=[es, my_lr_scheduler])


    # Use the model for prediction
    y_train_pred = mod.predict(X_train_scaled)
    y_test_pred = mod.predict(X_test_scaled)
    
    # Compute and store metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    print(f"Train R2: {train_r2} \n Test R2: {test_r2}")
    print(f"Train RMSE: {train_rmse} \n Test RMSE: {test_rmse}")

    results.append({
        'year': year,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse
    })



results





# Checkerboard CV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout

def assign_blocks(df, lat_col, lon_col, n_blocks=5):
    df['lat_block'] = pd.cut(df[lat_col], bins=n_blocks, labels=False)
    df['lon_block'] = pd.cut(df[lon_col], bins=n_blocks, labels=False)
    df['block_label'] = df['lat_block'].astype(int) + df['lon_block'].astype(int)
    df['block_color'] = df['block_label'].apply(lambda x: 'black' if x % 2 == 0 else 'white')
    return df

es = EarlyStopping(monitor='val_loss', patience=3, verbose=0,
    restore_best_weights=True)

# Set adaptive learning rate
my_lr_scheduler = LearningRateScheduler(adapt_learning_rate)

# mdat = mdat.assign(year = pd.to_datetime(mdat['date']).dt.year)

# rm_dates = ["2019-03-29", "2019-06-04", "2019-07-03", "2019-07-16", "2018-04-25"]

# moddat = mdat[~mdat['date'].isin(rm_dates)]

# moddat = moddat[moddat['year'] <= 2020]

# moddat = moddat[moddat['month'] <= "3"]

X = mdat[['year', 'snow', 'tmean', 'ppt', 'month', 'doy', 
          'lat', 'lon', 'lat_x_lon', 'elevation', 'landcover']]

y = mdat[['SWE']]

months = pd.get_dummies(X['month'])
landcover = pd.get_dummies(X['landcover'])

X = pd.concat([X, months, landcover], axis=1)
X = X.drop(columns=['month', 'landcover'])

# Sample Data
# Replace with your actual data
X = assign_blocks(X, 'lat', 'lon', n_blocks=10)

results = []
outdat_ = []

for color in ['black', 'white']:
    X_train = X[(X['block_color'] != color)]
    y_train = y[(X['block_color'] != color)]
    X_test = X[(X['block_color'] == color)]
    y_test = y[(X['block_color'] == color)]

    X_train = X_train.drop(columns=['block_label', 'block_color', 'lat_block', 'lon_block', 'year'])
    X_test = X_test.drop(columns=['block_label', 'block_color', 'lat_block', 'lon_block', 'year'])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    mod = Sequential()
    mod.add(Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)))
    mod.add(BatchNormalization())

    mod.add(Dense(32, activation='relu', kernel_regularizer=l2(0.005)))
    mod.add(BatchNormalization())

    mod.add(Dense(16, activation='relu', kernel_regularizer=l2(0.005)))
    mod.add(BatchNormalization())
    mod.add(Dropout(0.4))

    mod.add(Dense(8, activation='relu'))
    mod.add(BatchNormalization())
    mod.add(Dropout(0.3))

    mod.add(Dense(1, activation='linear'))

    mod.compile(optimizer='Adam', loss='mean_squared_error')

    mod.fit(X_train_scaled, y_train.values, validation_split=0.2, 
            shuffle=True, epochs=1000, batch_size=1024*2, 
            callbacks=[es, my_lr_scheduler])

    # Use the model for prediction
    y_train_pred = mod.predict(X_train_scaled)
    y_test_pred = mod.predict(X_test_scaled)
    
    # Compute and store metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    outdat = X_test.assign(color=color, 
                           y_pred = y_test_pred.ravel(),
                           y_true = y_test)

    outdat['residuals'] = outdat['y_true'] - outdat['y_pred']

    outdat_.append(outdat)
    
    results.append({
        'color': color,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse
    })

    print(results)

    # For now, just printing out train and test sizes
    print(f"Block Color: {color}, Train Size: {len(X_train)}, Test Size: {len(X_test)}")


model_dat = pd.concat(outdat_)
# model_dat.columns = model_dat.columns.astype(str)

model_dat = model_dat[['lat', 'lon', 'color', 'elevation', 'y_pred', 'y_true', 'residuals']]
model_dat.to_csv("data/Tuolumne_Watershed/checkboard_model_dat_output.csv")

# model_dat.to_parquet("data/Tuolumne_Watershed/checkboard_model_dat_output.parquet", compression=False)


# New traiing data
[{'color': 'black', 
'train_r2': 0.7088762709308165, 
'test_r2': 0.6929259169039914, 
'train_rmse': 0.26195376081778077, 
'test_rmse': 0.27084425478823365}, 

{'color': 'white', 
'train_r2': 0.6865292500835432, 
'test_r2': 0.672048067380516, 
'train_rmse': 0.27365069610293, 
'test_rmse': 0.2780295343811593}]



# With year
{'color': 'black', 
'train_r2': 0.742528272406793, 
'test_r2': 0.7213696989499503, 
'train_rmse': 0.22323010078645109, 
'test_rmse': 0.23442457802398842}

{'color': 'white',
'train_r2': 0.7338130345672325, 
'test_r2': 0.7275029986348159, 
'train_rmse': 0.22913021544425263, 
'test_rmse': 0.2296512668850365}

# Without year
{'color': 'black', 
'train_r2': 0.7003115357119869, 
'test_r2': 0.6812397905280756, 
'train_rmse': 0.24083688142413087, 
'test_rmse': 0.25073849951831945}, 

{'color': 'white', 
'train_r2': 0.6771024041984505, 
'test_r2': 0.6595499378133964, 
'train_rmse': 0.25236049845247477, 
'test_rmse': 0.2566933930956203}


# Without lat, lon, lat_x_lon
{'color': 'black', 
'train_r2': 0.6656127351239924, 
'test_r2': 0.6548911222869563, 
'train_rmse': 0.25439750420384294, 
'test_rmse': 0.26089576924993363}

[{'color': 'black', 
'train_r2': 0.6656127351239924, 
'test_r2': 0.6548911222869563, 
'train_rmse': 0.25439750420384294, 
'test_rmse': 0.26089576924993363},

{'color': 'white', 
'train_r2': 0.6654963059518864,
'test_r2': 0.655777317526288, 
'train_rmse': 0.2568558313844482, 
'test_rmse': 0.2581117195654205}


# --------------------------------------------------
# Train full model for predictions
X = mdat[['snow', 'tmean', 'ppt', 'month', 'doy', 
          'lat', 'lon', 'lat_x_lon', 'elevation', 'landcover']]

# X = moddat[['snow', 'month', 'lat_x', 'lon_x']]
y = mdat[['SWE']]

months = pd.get_dummies(X['month'])
landcover = pd.get_dummies(X['landcover'])

X = pd.concat([X, months, landcover], axis=1)
X = X.drop(columns=['month', 'landcover'])

X_train = X
y_train = y

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

mod = Sequential()
mod.add(Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)))
mod.add(BatchNormalization())

mod.add(Dense(32, activation='relu', kernel_regularizer=l2(0.005)))
mod.add(BatchNormalization())

mod.add(Dense(16, activation='relu', kernel_regularizer=l2(0.005)))
mod.add(BatchNormalization())
mod.add(Dropout(0.4))

mod.add(Dense(8, activation='relu'))
mod.add(BatchNormalization())
mod.add(Dropout(0.3))

mod.add(Dense(1, activation='linear'))

mod.compile(optimizer='Adam', loss='mean_squared_error')

mod.fit(X_train_scaled, y_train.values, validation_split=0.2, 
        shuffle=True, epochs=1000, batch_size=1024*2, 
        callbacks=[es, my_lr_scheduler])

# Use the model for prediction
y_train_pred = mod.predict(X_train_scaled)

# Compute and store metrics
train_r2 = r2_score(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

print(f"Train R2: {train_r2} \ RMSE: {train_rmse}")

# In [142]: print(f"Train R2: {train_r2} \ RMSE: {train_rmse}")
# Train R2: 0.671780051330245 \ RMSE: 0.25328199022197384

import joblib

# Save the Keras model
mod.save('data/Tuolumne_Watershed/models/NN-ASO-SWE-Model_V2.h5')

# Save the Scaler object
joblib.dump(scaler, 'data/Tuolumne_Watershed/models/NN-ASO-SWE-Scaler_V2.pkl')
