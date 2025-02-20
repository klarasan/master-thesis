from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from keras import layers

plt.rcParams['font.family'] = 'Times New Roman'

coral = (1,0.5,0.31,0.7)
orange = (1, 0.44, 0, 1)
thistle = (0.9,0.75,0.9,1)
plum = (0.5, 0, 0.5, 1)
slate = (72/255,61/255,139/255, 1)
#colors = [coral, orange, thistle, plum, slate]
colors = [orange, plum, slate]

def num_features():
    vars14 = ['aet', 'def', 'pet', 'ppt', 'q', 'soil', 'srad', 'swe', 'tmax', 'tmin', 'vap', 'ws', 'vpd', 'PDSI']
    vars10 = ['aet', 'def', 'ppt', 'q', 'soil', 'srad', 'tmax', 'vap', 'vpd', 'PDSI']
    vars8 = ['tmax', 'vpd', 'def', 'soil', 'ppt', 'PDSI', 'srad', 'q']
    vars6 = ['tmax', 'vpd', 'def', 'soil', 'ppt', 'PDSI']
    vars3 = ['srad', 'def', 'PDSI']
    vars_list = [vars14, vars10, vars8, vars6, vars3]

    df = pd.read_csv('data/avg_std_12_years_bilinear_interp_w_outliers.csv', on_bad_lines='skip')

    for v in range(0, 5):
        val_accuracies = []
        vars = vars_list[v]
        for num_years in range(2, 13, 2):
            print(f'starting yearly training with {num_years} years')
            times = range(1-num_years, 1)
            X = np.zeros((len(df), len(vars), len(times))) 

            # Populate X with values from x
            for i, var in enumerate(vars):
                for j, t in enumerate(times):
                    col_name = f"{var}_year{t}_mean"
                    X[:, i, j] = df[col_name].values
            X = np.transpose(X, (0, 2, 1))
            
            ref_ids = df["Ref_ID"].values
            labels = df["label"].values 

            preds = []
            values = []

            for id in range(0, 153):
                X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []

                # Boolean masks for selecting train/test samples
                train_mask = (ref_ids != id)
                test_mask = (ref_ids == id)

                X_train_list.append(X[train_mask])
                y_train_list.append(labels[train_mask])

                X_test_list.append(X[test_mask])
                y_test_list.append(labels[test_mask])

                # Convert lists to NumPy arrays
                X_train = np.concatenate(X_train_list, axis=0)
                y_train = np.concatenate(y_train_list, axis=0)

                X_test = np.concatenate(X_test_list, axis=0)
                y_test = np.concatenate(y_test_list, axis=0)

                if y_test.shape[0] > 0:
                    print(f'Sample: {id}. {num_years} years and {len(vars)} variables')
                    model = keras.Sequential([
                    layers.LSTM(128, return_sequences=True, input_shape=(num_years, len(vars))),
                    layers.Dropout(0.3),
                    layers.LSTM(64),
                    layers.Dropout(0.3),
                    layers.Dense(32, activation="relu"),
                    layers.Dense(1, activation="sigmoid")  
                    ])
                    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
                    _ = model.fit(X_train, y_train, epochs=50, batch_size=32)
                    y_pred = model.predict(X_test)
                    y_pred_labels = (y_pred > 0.5).astype(int)

                    values.extend(y_test)
                    preds.extend(y_pred_labels)
                        
            val_acc = accuracy_score(values, preds)
            val_acc = round(val_acc * 100.0, 2)
            print(val_acc)
            val_accuracies.append(val_acc)
        plt.plot(range(2, 13, 2), val_accuracies, label=f'{len(vars)} input channels', color=colors[v])

    plt.xlabel('Number of input years')
    plt.ylabel('Test accuracy')
    plt.ylim(0, 100)
    plt.title('Multi-layer perceptron')
    plt.legend()
    plt.grid()
    plt.show()
    return

def lstm_vs_gru():
    vars = ['tmax', 'vpd', 'def', 'soil', 'ppt', 'PDSI', 'srad', 'q']
    df = pd.read_csv('data/avg_std_12_years_bilinear_interp_w_outliers.csv', on_bad_lines='skip')

    lstm_val_accuracies = []
    gru_val_accuracies = []
    for num_years in range(4, 13, 4):
        times = range(1-num_years, 1)
        X = np.zeros((len(df), len(vars), len(times))) 

        # Populate X with values from x
        for i, var in enumerate(vars):
            for j, t in enumerate(times):
                col_name = f"{var}_year{t}_mean"
                X[:, i, j] = df[col_name].values
        X = np.transpose(X, (0, 2, 1))
        
        ref_ids = df["Ref_ID"].values
        labels = df["label"].values 

        lstm_preds = []
        gru_preds = []
        values = []

        for id in range(0, 153, 2):
            id0 = id
            id1 = id+1

            X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []

            # Boolean masks for selecting train/test samples
            train_mask = ~((ref_ids == id0) | (ref_ids == id1))
            test_mask = (ref_ids == id0) | (ref_ids == id1)

            X_train_list.append(X[train_mask])
            y_train_list.append(labels[train_mask])

            X_test_list.append(X[test_mask])
            y_test_list.append(labels[test_mask])

            # Convert lists to NumPy arrays
            X_train = np.concatenate(X_train_list, axis=0)
            y_train = np.concatenate(y_train_list, axis=0)

            X_test = np.concatenate(X_test_list, axis=0)
            y_test = np.concatenate(y_test_list, axis=0)

            if y_test.shape[0] > 0:
                print(f'Sample: {id0} and {id1}. {num_years} years and {len(vars)} variables')
                values.extend(y_test)

                lstm_model = keras.Sequential([
                layers.LSTM(128, return_sequences=True, input_shape=(num_years, len(vars))),
                layers.Dropout(0.3),
                layers.LSTM(64),
                layers.Dropout(0.3),
                layers.Dense(32, activation="relu"),
                layers.Dense(1, activation="sigmoid")  
                ])
                lstm_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
                _ = lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
                lstm_y_pred = lstm_model.predict(X_test)
                lstm_y_pred_labels = (lstm_y_pred > 0.5).astype(int)
                lstm_preds.extend(lstm_y_pred_labels)

                gru_model = keras.Sequential([
                layers.GRU(128, return_sequences=True, input_shape=(num_years, len(vars))),
                layers.Dropout(0.3),
                layers.GRU(64),
                layers.Dropout(0.3),
                layers.Dense(32, activation="relu"),
                layers.Dense(1, activation="sigmoid")  
                ])
                gru_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
                _ = gru_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
                gru_y_pred = gru_model.predict(X_test)
                gru_y_pred_labels = (gru_y_pred > 0.5).astype(int)
                gru_preds.extend(gru_y_pred_labels)
                    
        lstm_val_acc = accuracy_score(values, lstm_preds)
        lstm_val_acc = round(lstm_val_acc * 100.0, 2)
        print(f'LSTM accuracy: {lstm_val_acc}')
        lstm_val_accuracies.append(lstm_val_acc)

        gru_val_acc = accuracy_score(values, gru_preds)
        gru_val_acc = round(gru_val_acc * 100.0, 2)
        print(f'GRU accuracy: {gru_val_acc}')
        gru_val_accuracies.append(gru_val_acc)

    plt.plot(range(4, 13, 4), lstm_val_accuracies, label=f'LSTM model', color=colors[1])
    plt.plot(range(4, 13, 4), gru_val_accuracies, label=f'GRU model', color=colors[3])

    plt.xlabel('Number of input years')
    plt.ylabel('Test accuracy')
    plt.ylim(0, 100)
    plt.title('LSTM vs GRU')
    plt.legend()
    plt.grid()
    plt.show()
    return

def monthly_vs_yearly():
    vars = ['tmax', 'vpd', 'def', 'soil', 'ppt', 'PDSI', 'srad', 'q']
    df = pd.read_csv('data/12_years_bilinear_interp_w_outliers.csv', on_bad_lines='skip')

    val_accuracies = []
    for num_years in range(2, 13, 2):
        years = range(1-num_years, 1)
        months = range(1, 13)
        X = np.zeros((len(df), len(vars), len(years) * len(months)))  

        # Populate X with values from df
        for i, var in enumerate(vars):
            for j, year in enumerate(years):
                for k, month in enumerate(months):
                    col_name = f"{var}_year{year}_month{month}"
                    time_idx = j * len(months) + k
                    X[:, i, time_idx] = df[col_name].values
        X = np.transpose(X, (0, 2, 1))

        ref_ids = df["Ref_ID"].values
        labels = df["label"].values 

        preds = []
        values = []

        for id in range(0, 153, 2):
            id0 = id
            id1 = id+1

            X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []

            # Boolean masks for selecting train/test samples
            train_mask = ~((ref_ids == id0) | (ref_ids == id1))
            test_mask = (ref_ids == id0) | (ref_ids == id1)

            X_train_list.append(X[train_mask])
            y_train_list.append(labels[train_mask])

            X_test_list.append(X[test_mask])
            y_test_list.append(labels[test_mask])

            # Convert lists to NumPy arrays
            X_train = np.concatenate(X_train_list, axis=0)
            y_train = np.concatenate(y_train_list, axis=0)

            X_test = np.concatenate(X_test_list, axis=0)
            y_test = np.concatenate(y_test_list, axis=0)

            if y_test.shape[0] > 0:
                print(f'Sample: {id0} and {id1}. {num_years} years and monthly data')
                values.extend(y_test)

                model = keras.Sequential([
                layers.LSTM(128, return_sequences=True, input_shape=(num_years*12, len(vars))),
                layers.Dropout(0.3),
                layers.LSTM(64),
                layers.Dropout(0.3),
                layers.Dense(32, activation="relu"),
                layers.Dense(1, activation="sigmoid")  
                ])
                model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
                _ = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
                y_pred = model.predict(X_test)
                y_pred_labels = (y_pred > 0.5).astype(int)
                preds.extend(y_pred_labels)

        val_acc = accuracy_score(values, preds)
        val_acc = round(val_acc * 100.0, 2)
        print(f'Accuracy: {val_acc}')
        val_accuracies.append(val_acc)
    plt.plot(range(2, 13, 2), val_accuracies, label='Monthly data', color=colors[1])

    df = pd.read_csv('data/avg_std_12_years_bilinear_interp_w_outliers.csv', on_bad_lines='skip')

    val_accuracies = []
    for num_years in range(2, 13, 2):
        years = range(1-num_years, 1)
        X = np.zeros((len(df), len(vars), len(years))) 

        # Populate X with values from x
        for i, var in enumerate(vars):
            for j, t in enumerate(years):
                col_name = f"{var}_year{t}_mean"
                X[:, i, j] = df[col_name].values
        X = np.transpose(X, (0, 2, 1))
        
        ref_ids = df["Ref_ID"].values
        labels = df["label"].values 

        preds = []
        values = []

        for id in range(0, 153, 2):
            id0 = id
            id1 = id+1

            X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []

            # Boolean masks for selecting train/test samples
            train_mask = ~((ref_ids == id0) | (ref_ids == id1))
            test_mask = (ref_ids == id0) | (ref_ids == id1)

            X_train_list.append(X[train_mask])
            y_train_list.append(labels[train_mask])

            X_test_list.append(X[test_mask])
            y_test_list.append(labels[test_mask])

            # Convert lists to NumPy arrays
            X_train = np.concatenate(X_train_list, axis=0)
            y_train = np.concatenate(y_train_list, axis=0)

            X_test = np.concatenate(X_test_list, axis=0)
            y_test = np.concatenate(y_test_list, axis=0)

            if y_test.shape[0] > 0:
                print(f'Sample: {id0} and {id1}. {num_years} years and yearly data')
                values.extend(y_test)

                model = keras.Sequential([
                layers.LSTM(128, return_sequences=True, input_shape=(num_years*12, len(vars))),
                layers.Dropout(0.3),
                layers.LSTM(64),
                layers.Dropout(0.3),
                layers.Dense(32, activation="relu"),
                layers.Dense(1, activation="sigmoid")  
                ])
                model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
                _ = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
                y_pred = model.predict(X_test)
                y_pred_labels = (y_pred > 0.5).astype(int)
                preds.extend(y_pred_labels)

        val_acc = accuracy_score(values, preds)
        val_acc = round(val_acc * 100.0, 2)
        print(f'Accuracy: {val_acc}')
        val_accuracies.append(val_acc)
    plt.plot(range(2, 13, 2), val_accuracies, label='Yearly data', color=colors[3])

    plt.xlabel('Number of input years')
    plt.ylabel('Test accuracy')
    plt.ylim(0, 100)
    plt.title('LSTM')
    plt.legend()
    plt.grid()
    plt.show()
    return

def num_channels():
    vars14 = ['aet', 'def', 'pet', 'ppt', 'q', 'soil', 'srad', 'swe', 'tmax', 'tmin', 'vap', 'ws', 'vpd', 'PDSI']
    vars7 = ['tmax', 'vpd', 'def', 'soil', 'ppt', 'PDSI', 'srad']
    vars3 = ['srad', 'ppt', 'PDSI']
    vars_list = [vars14, vars7, vars3]

    df = pd.read_csv('data/12_years_bilinear_interp_w_outliers.csv', on_bad_lines='skip')

    for v in range(0, 3):
        val_accuracies = []
        vars = vars_list[v]
        for num_years in range(3, 6):
            years = range(1-num_years, 1)
            months = range(1, 13)
            X = np.zeros((len(df), len(vars), len(years) * len(months)))  

            # Populate X with values from df
            for i, var in enumerate(vars):
                for j, year in enumerate(years):
                    for k, month in enumerate(months):
                        col_name = f"{var}_year{year}_month{month}"
                        time_idx = j * len(months) + k
                        X[:, i, time_idx] = df[col_name].values
            X = np.transpose(X, (0, 2, 1))

            ref_ids = df["Ref_ID"].values
            labels = df["label"].values 

            preds = []
            values = []

            for id in range(0, 153, 2):
                id0 = id
                id1 = id+1

                X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []

                # Boolean masks for selecting train/test samples
                train_mask = ~((ref_ids == id0) | (ref_ids == id1))
                test_mask = (ref_ids == id0) | (ref_ids == id1)

                X_train_list.append(X[train_mask])
                y_train_list.append(labels[train_mask])

                X_test_list.append(X[test_mask])
                y_test_list.append(labels[test_mask])

                # Convert lists to NumPy arrays
                X_train = np.concatenate(X_train_list, axis=0)
                y_train = np.concatenate(y_train_list, axis=0)

                X_test = np.concatenate(X_test_list, axis=0)
                y_test = np.concatenate(y_test_list, axis=0)

                if y_test.shape[0] > 0:
                    print(f'Sample: {id0} and {id1}. {num_years} years and {len(vars)} variables')
                    values.extend(y_test)

                    model = keras.Sequential([
                    layers.LSTM(128, return_sequences=True, input_shape=(num_years*12, len(vars))),
                    layers.Dropout(0.3),
                    layers.LSTM(64),
                    layers.Dropout(0.3),
                    layers.Dense(32, activation="relu"),
                    layers.Dense(1, activation="sigmoid")  
                    ])
                    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
                    _ = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
                    y_pred = model.predict(X_test)
                    y_pred_labels = (y_pred > 0.5).astype(int)
                    preds.extend(y_pred_labels)

            val_acc = accuracy_score(values, preds)
            val_acc = round(val_acc * 100.0, 2)
            print(f'Accuracy: {val_acc}')
            val_accuracies.append(val_acc)

        plt.plot(range(3, 6), val_accuracies, label=f'{len(vars)} input channels', color=colors[v])

    plt.xlabel('Number of input years')
    plt.xticks(np.arange(3, 6, step=1))
    plt.ylabel('Test accuracy')
    plt.ylim(0, 100)
    plt.title('LSTM')
    plt.legend()
    plt.grid()
    plt.show()
    return

if __name__ == "__main__":
    # lstm_vs_gru()
    # monthly_vs_yearly()
    # num_features()
    num_channels()