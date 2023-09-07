import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.metrics import r2_score, mean_squared_error
from keras.regularizers import l2


def adapt_learning_rate(epoch):
    return 0.001 * epoch

# -----------------------------------
# Build model data

mdat = pd.read_parquet("data/Tuolumne_Watershed/model_data_elevation_prism_sinceSep_nlcd.parquet")
mdat = mdat.dropna()

mdat = mdat.assign(year = pd.to_datetime(mdat['date']).dt.strftime("%Y"),
    month = pd.to_datetime(mdat['date']).dt.strftime("%m"),
    doy = pd.to_datetime(mdat['date']).dt.strftime("%j"))

mdat.head()

mdat['lat_x_lon'] = mdat['lat_x'] * mdat['lon_x']


mdat.columns

X = mdat[['snow', 'tmean', 'ppt', 'month', 'doy',
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

# Set early stop
es = EarlyStopping(monitor='val_loss', patience=5, verbose=0,
    restore_best_weights=True)

# Set adaptive learning rate
my_lr_scheduler = LearningRateScheduler(adapt_learning_rate)

mdat = mdat.assign(year = pd.to_datetime(mdat['date']).dt.year)

rm_dates = ["2019-03-29", "2019-06-04", "2019-07-03", "2019-07-16", "2018-04-25"]

moddat = mdat[~mdat['date'].isin(rm_dates)]

# moddat = moddat[moddat['year'] <= 2020]

# moddat = moddat[moddat['month'] <= "3"]

X = moddat[['year', 'snow', 'tmean', 'tmax', 'tmin', 'ppt', 'month', 
          'lat', 'lon', 'lat_x_lon', 'elevation', 'landcover']]

X = moddat[['year', 'snow', 'tmean', 'ppt', 'month', 'lat', 'lon',
          'elevation', 'landcover']]

X = moddat[['snow', 'tmean', 'ppt', 'month', 'doy', 'year',
          'lat', 'lon', 'lat_x_lon', 'elevation', 'landcover']]

# X = moddat[['snow', 'month', 'lat_x', 'lon_x']]
y = moddat[['SWE']]

months = pd.get_dummies(X['month'])
landcover = pd.get_dummies(X['landcover'])

# X = pd.concat([X, months], axis=1)
# X = X.drop(columns=['month'])

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


    mod = Sequential()
    mod.add(Dense(15, activation='relu', kernel_regularizer=l2(0.001)))
    mod.add(Dropout(0.75))
    mod.add(Dense(10, activation='relu', kernel_regularizer=l2(0.001)))
    mod.add(Dropout(0.75))
    mod.add(Dense(5, activation='relu'))
    mod.add(Dense(1, activation='linear'))
    mod.compile(optimizer='Adam', loss='mean_squared_error')
    mod.fit(X_train_scaled, y_train.values, validation_split=0.2,
        shuffle=True, epochs=1000,  batch_size=1024*2,
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




# mod = Sequential()
# mod.add(Dense(15, activation='relu', kernel_regularizer=l2(0.01)))
# mod.add(Dropout(0.75))
# mod.add(Dense(10, activation='relu', kernel_regularizer=l2(0.01)))
# mod.add(Dropout(0.75))
# mod.add(Dense(5, activation='relu'))
# mod.add(Dense(1, activation='linear'))
# mod.compile(optimizer='Adam', loss='mean_squared_error')
# mod.fit(X_train_scaled, y_train.values, validation_split=0.2,
#     shuffle=True, epochs=1000,  batch_size=1024*2,
#     callbacks=[es, my_lr_scheduler]
#     )
[{'year': 2013,
  'train_r2': 0.44848155270575385,
  'test_r2': 0.4032524271496962,
  'train_rmse': 0.3376569320871912,
  'test_rmse': 0.20178461665945713},
 {'year': 2016,
  'train_r2': 0.5803549051693816,
  'test_r2': 0.40648843860082984,
  'train_rmse': 0.27931592326544336,
  'test_rmse': 0.35527968621224976},
 {'year': 2017,
  'train_r2': 0.1642706017012926,
  'test_r2': 0.1408199978475846,
  'train_rmse': 0.39552264945733656,
  'test_rmse': 0.4749628916592649},
 {'year': 2018,
  'train_r2': 0.5820092466796385,
  'test_r2': 0.40002559680903504,
  'train_rmse': 0.2877711170298163,
  'test_rmse': 0.28666454447473405},
 {'year': 2019,
  'train_r2': 0.5365786818112213,
  'test_r2': 0.26525176968687414,
  'train_rmse': 0.24218147328671233,
  'test_rmse': 0.6230759262707221},
 {'year': 2020,
  'train_r2': 0.276505310398825,
  'test_r2': 0.29130181824286006,
  'train_rmse': 0.39385664771641277,
  'test_rmse': 0.20940731345311794},
 {'year': 2021,
  'train_r2': 0.47532905511354984,
  'test_r2': -0.06093299713924738,
  'train_rmse': 0.33913505735392274,
  'test_rmse': 0.2087387850838748},
 {'year': 2022,
  'train_r2': 0.5045006040529805,
  'test_r2': 0.4406196458153937,
  'train_rmse': 0.3317226239581338,
  'test_rmse': 0.16661753995532996}]






mod = Sequential()
mod.add(Dense(15, activation='relu', kernel_regularizer=l2(0.001)))
mod.add(Dropout(0.75))
mod.add(Dense(10, activation='relu', kernel_regularizer=l2(0.001)))
mod.add(Dropout(0.75))
mod.add(Dense(5, activation='relu'))
mod.add(Dense(1, activation='linear'))
mod.compile(optimizer='Adam', loss='mean_squared_error')
mod.fit(X_train_scaled, y_train.values, validation_split=0.2,
    shuffle=True, epochs=1000,  batch_size=1024*2,
    callbacks=[es, my_lr_scheduler]
    )








# Old results
[{'year': 2013,
  'train_r2': 0.7118913220853421,
  'test_r2': -0.07311908089421215,
  'train_rmse': 0.2440800621978105,
  'test_rmse': 0.2705930774966701},
 {'year': 2016,
  'train_r2': 0.6727196582119688,
  'test_r2': 0.05669728140349484,
  'train_rmse': 0.24672947061271175,
  'test_rmse': 0.44790027386427367},
 {'year': 2017,
  'train_r2': 0.6898285404161429,
  'test_r2': -0.6434658031247538,
  'train_rmse': 0.24100673479430965,
  'test_rmse': 0.6568979943371933},
 {'year': 2018,
  'train_r2': 0.7034415687827724,
  'test_r2': 0.47609418960970495,
  'train_rmse': 0.24244932505733363,
  'test_rmse': 0.2675518326284094},
 {'year': 2019,
  'train_r2': 0.6751431134487768,
  'test_r2': 0.3908936800200018,
  'train_rmse': 0.2027559461989302,
  'test_rmse': 0.566975308485237},
 {'year': 2020,
  'train_r2': 0.7298617260456821,
  'test_r2': 0.41581833860946893,
  'train_rmse': 0.24069394354732584,
  'test_rmse': 0.19012321323999773},
 {'year': 2021,
  'train_r2': 0.6619219811815906,
  'test_r2': 0.26224906629300937,
  'train_rmse': 0.27225995270207004,
  'test_rmse': 0.1740660430365345},
 {'year': 2022,
  'train_r2': 0.657793637836098,
  'test_r2': -1.237407082458354,
  'train_rmse': 0.27570225657431013,
  'test_rmse': 0.3332265658737885}]




# Spatial Blocking
def assign_blocks(df, lat_col, lon_col, n_blocks=5):
    # Create block labels
    df['lat_block'] = pd.cut(df[lat_col], bins=n_blocks, labels=False)
    df['lon_block'] = pd.cut(df[lon_col], bins=n_blocks, labels=False)
    df['block_label'] = df['lat_block'].astype(str) + "_" + df['lon_block'].astype(str)
    return df

X = assign_blocks(X, 'lat', 'lon', n_blocks=5)

results = []
unique_years = sorted(X['year'].unique())
for year in unique_years:
    # Create blocks for this year
    X_year = X[X['year'] == year]
    X_year = assign_blocks(X_year, 'lat', 'lon', n_blocks=20)
    
    unique_blocks = X_year['block_label'].unique()

    for block in unique_blocks:
        # Filter data by year and block
        X_train = X[(X['year'] != year) | (X['block_label'] != block)]
        y_train = y[(X['year'] != year) | (X['block_label'] != block)]
        X_test = X[(X['year'] == year) & (X['block_label'] == block)]
        y_test = y[(X['year'] == year) & (X['block_label'] == block)]

        # Drop year and block label columns
        X_train = X_train.drop(columns=['year', 'block_label'])
        X_test = X_test.drop(columns=['year', 'block_label'])
        
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

        # Your neural network model
        # mod = Sequential()
        # mod.add(Dense(15, activation='relu', kernel_regularizer=l2(0.01)))
        # mod.add(Dropout(0.75))
        # mod.add(Dense(10, activation='relu', kernel_regularizer=l2(0.01)))
        # mod.add(Dropout(0.75))
        # mod.add(Dense(5, activation='relu'))
        # mod.add(Dense(1, activation='linear'))
        # mod.compile(optimizer='Adam', loss='mean_squared_error')

        mod = Sequential()
        mod.add(Dense(15, activation='relu', kernel_regularizer=l2(0.01)))
        mod.add(Dropout(0.75))
        mod.add(Dense(10, activation='relu', kernel_regularizer=l2(0.01)))
        mod.add(Dropout(0.75))
        mod.add(Dense(5, activation='relu'))
        mod.add(Dense(1, activation='linear'))
        mod.compile(optimizer='Adam', loss='mean_squared_error')
        
        # Train the model
        mod.fit(X_train_scaled, y_train.values, validation_split=0.2,
            shuffle=True, epochs=1000,  batch_size=1024*2,
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
            'block': block,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse
        })

        print(results)




# Checkerboard CV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
# Your other imports for es, my_lr_scheduler, etc.

def assign_blocks(df, lat_col, lon_col, n_blocks=5):
    df['lat_block'] = pd.cut(df[lat_col], bins=n_blocks, labels=False)
    df['lon_block'] = pd.cut(df[lon_col], bins=n_blocks, labels=False)
    df['block_label'] = df['lat_block'].astype(int) + df['lon_block'].astype(int)
    df['block_color'] = df['block_label'].apply(lambda x: 'black' if x % 2 == 0 else 'white')
    return df

# Sample Data
# Replace with your actual data

X = assign_blocks(X, 'lat', 'lon', n_blocks=20)
results = []

for color in ['black', 'white']:
    X_train = X[(X['block_color'] != color)]
    y_train = y[(X['block_color'] != color)]
    X_test = X[(X['block_color'] == color)]
    y_test = y[(X['block_color'] == color)]

    X_train = X_train.drop(columns=['block_label', 'block_color', 'lat_block', 'lon_block'])
    X_test = X_test.drop(columns=['block_label', 'block_color', 'lat_block', 'lon_block'])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    mod = Sequential()
    mod.add(Dense(15, activation='relu', kernel_regularizer=l2(0.01)))
    mod.add(Dropout(0.75))
    mod.add(Dense(10, activation='relu', kernel_regularizer=l2(0.01)))
    mod.add(Dropout(0.75))
    mod.add(Dense(5, activation='relu'))
    mod.add(Dense(1, activation='linear'))
    mod.compile(optimizer='Adam', loss='mean_squared_error')
    
    # Train the model
    mod.fit(X_train_scaled, y_train.values, validation_split=0.2,
        shuffle=True, epochs=1000,  batch_size=1024*2,
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
        'block': block,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse
    })

    print(results)


    # For now, just printing out train and test sizes
    print(f"Block Color: {color}, Train Size: {len(X_train)}, Test Size: {len(X_test)}")






