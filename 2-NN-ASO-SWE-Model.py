from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from keras.callbacks import LearningRateScheduler, EarlyStopping
import joblib
import json


def assign_blocks(df, lat_col, lon_col, n_blocks=5):
    df['lat_block'] = pd.cut(df[lat_col], bins=n_blocks, labels=False)
    df['lon_block'] = pd.cut(df[lon_col], bins=n_blocks, labels=False)
    df['block_label'] = df['lat_block'].astype(int) + df['lon_block'].astype(int)
    df['block_color'] = df['block_label'].apply(lambda x: 'black' if x % 2 == 0 else 'white')
    return df


def adapt_learning_rate(epoch):
    return 0.001 * epoch


es = EarlyStopping(monitor='val_loss', patience=3, verbose=0,
    restore_best_weights=True)


my_lr_scheduler = LearningRateScheduler(adapt_learning_rate)


def checkboard_cv_model(watershed):

    mdat = pd.read_parquet(f"data/{watershed}/processed/model_data_elevation_prism_sinceSep_nlcd.parquet")
    
    # mdat = mdat.assign(year = pd.to_datetime(mdat['date']).dt.strftime("%Y"),
    #     month = pd.to_datetime(mdat['date']).dt.strftime("%m"),
    #     doy = pd.to_datetime(mdat['date']).dt.strftime("%j"))

    # Convert the 'date' column to datetime just once
    mdat['date'] = pd.to_datetime(mdat['date'])

    # Extract the year, month, and day of the year directly from the datetime object
    mdat = mdat.assign(
        year=mdat['date'].dt.year.astype(str),
        month=mdat['date'].dt.month.astype(str).str.zfill(2),
        doy=mdat['date'].dt.dayofyear.astype(str).str.zfill(3)
    )

    # mdat = mdat[mdat['year'].isin(["2021", "2022", "2023"])].reset_index(drop=True)

    X = mdat[['year', 'snow', 'tmean', 'ppt', 'month', 'doy', 
              'lat', 'lon', 'lat_x_lon', 'elevation', 'slope', 'aspect', 
              'landcover']]

    y = np.log(1 + mdat['SWE'])

    months = pd.get_dummies(X['month'], prefix='month')
    landcover = pd.get_dummies(X['landcover'], prefix='landcover')

    X = pd.concat([X, months, landcover], axis=1)
    X = X.drop(columns=['month', 'landcover'])

    # Get blocks
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

    #concat blocks
    model_dat = pd.concat(outdat_)

    # Convert from log(SWE) to SWE
    model_dat['y_pred'] = np.exp(model_dat['y_pred']) - 1
    model_dat['y_true'] = np.exp(model_dat['y_true']) - 1

    model_dat['y_true'].max()
    model_dat['y_pred'].max()

    meta_data = {'basin': watershed,
                 'cv_results': results}

    meta_data = {key: str(value) for key, value in meta_data.items()}

    # Save metadata
    with open(f"data/{watershed}/models/Model-MetaData.json", 'w') as file:
        json.dump(meta_data, file, indent=4)  # 'indent' for pretty printing
    
    # Save model data
    model_dat = model_dat[['lat', 'lon', 'color', 'elevation', 'y_pred', 'y_true', 'residuals']]
    model_dat.to_csv(f"data/{watershed}/processed/checkboard_model_dat_output.csv", index=False)


def train_full_model(watershed):

    mdat = pd.read_parquet(f"data/{watershed}/processed/model_data_elevation_prism_sinceSep_nlcd.parquet")

    # Convert the 'date' column to datetime objects once
    mdat['date'] = pd.to_datetime(mdat['date'])

    # Extract year, month, and day of year
    mdat = mdat.assign(
        year=mdat['date'].dt.year.astype(str),
        month=mdat['date'].dt.month.astype(str).str.zfill(2),
        doy=mdat['date'].dt.dayofyear.astype(str).str.zfill(3)
    )

    # --------------------------------------------------
    # Train full model for predictions
    X = mdat[['snow', 'tmean', 'ppt', 'month', 'doy', 
              'lat', 'lon', 'lat_x_lon', 'elevation', 'slope', 'aspect', 'landcover']]

    y = np.log(1 + mdat['SWE'])

    months = pd.get_dummies(X['month'], prefix="month")
    landcover = pd.get_dummies(X['landcover'], prefix="landcover")

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

    # Save column names
    column_names = X.columns.tolist()
    np.save(f"data/{watershed}/models/col_list.npy", column_names)

    # Save the Keras model
    mod.save(f"data/{watershed}/models/NN-ASO-SWE-Model.h5")

    # Save the Scaler object
    joblib.dump(scaler, f"data/{watershed}/models/NN-ASO-SWE-Scaler.pkl")




if __name__ == "__main__":

    watershed = "Tuolumne_Watershed"

    watershed = "Blue_Dillon_Watershed"

    watershed = "Dolores_Watershed"

    watershed = "Conejos_Watershed"

    checkboard_cv_model(watershed)    

    train_full_model(watershed)

    [checkboard_cv_model(x) for x in ['Tuolumne_Watershed', 
                                   'Blue_Dillon_Watershed', 
                                   'Dolores_Watershed', 
                                   'Conejos_Watershed']]


    [train_full_model(x) for x in ['Tuolumne_Watershed', 
                                   'Blue_Dillon_Watershed', 
                                   'Dolores_Watershed', 
                                   'Conejos_Watershed']]






