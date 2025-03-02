import re
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

def network_size():
    vars7 = ['tmax', 'vpd', 'def', 'soil', 'ppt', 'PDSI', 'srad']
    df = pd.read_csv('data/12_years_bilinear_interp_w_outliers.csv', on_bad_lines='skip')

    val_accuracies1 = []
    val_accuracies2 = []
    val_accuracies3 = []
    for num_years in range(3, 6):
        years = range(1-num_years, 1)
        months = range(1, 13)
        X = np.zeros((len(df), len(vars7), len(years) * len(months)))  

        # Populate X with values from df
        for i, var in enumerate(vars7):
            for j, year in enumerate(years):
                for k, month in enumerate(months):
                    col_name = f"{var}_year{year}_month{month}"
                    time_idx = j * len(months) + k
                    X[:, i, time_idx] = df[col_name].values
        X = np.transpose(X, (0, 2, 1))

        ref_ids = df["Ref_ID"].values
        labels = df["label"].values 

        preds1 = []
        preds2 = []
        preds3 = []
        values = []
        for id in range(0, 153, 3):
            id0 = id
            id1 = id+1
            id2 = id+2

            X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []

            # Boolean masks for selecting train/test samples
            train_mask = ~((ref_ids == id0) | (ref_ids == id1)| (ref_ids == id2))
            test_mask = (ref_ids == id0) | (ref_ids == id1) | (ref_ids == id2)

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
                print(f'Sample: {id0} and {id1} and {id2}. {num_years} years')
                print(f'{len(y_train)} training samples')
                print(f'{len(y_test)} testing samples')
                values.extend(y_test)

                model1 = keras.Sequential([
                layers.LSTM(64, return_sequences=True, input_shape=(num_years * 12, len(vars7))),  # Halved units
                layers.Dropout(0.2),  # Slightly reduced dropout
                layers.LSTM(32),  # Smaller second LSTM layer
                layers.Dropout(0.2),
                layers.Dense(16, activation="relu"),  # Fewer units in dense layer
                layers.Dense(1, activation="sigmoid")  # Same output for comparison
                ])
                model1.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
                _ = model1.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
                y_pred = model1.predict(X_test)
                y_pred_labels = (y_pred > 0.5).astype(int)
                preds1.extend(y_pred_labels)

                model2 = keras.Sequential([
                layers.LSTM(128, return_sequences=True, input_shape=(num_years*12, len(vars7))),
                layers.Dropout(0.3),
                layers.LSTM(64),
                layers.Dropout(0.3),
                layers.Dense(32, activation="relu"),
                layers.Dense(1, activation="sigmoid")  
                ])
                model2.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
                _ = model2.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
                y_pred = model2.predict(X_test)
                y_pred_labels = (y_pred > 0.5).astype(int)
                preds2.extend(y_pred_labels)

                model3 = keras.Sequential([
                layers.LSTM(256, return_sequences=True, input_shape=(num_years * 12, len(vars7))),  # Increased units
                layers.Dropout(0.4),  # Slightly higher dropout for regularization
                layers.LSTM(128, return_sequences=True),  # Added an extra LSTM layer
                layers.Dropout(0.4),
                layers.LSTM(64),  # Original second LSTM layer size
                layers.Dropout(0.3),
                layers.Dense(64, activation="relu"),  # Larger dense layer
                layers.Dense(32, activation="relu"),  # Added another dense layer for complexity
                layers.Dense(1, activation="sigmoid")
                ])
                model3.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
                _ = model3.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
                y_pred = model3.predict(X_test)
                y_pred_labels = (y_pred > 0.5).astype(int)
                preds3.extend(y_pred_labels)

        val_acc1 = accuracy_score(values, preds1)
        val_acc1 = round(val_acc1 * 100.0, 2)
        print(f'Accuracy model 1: {val_acc1}')
        val_accuracies1.append(val_acc1)

        val_acc2 = accuracy_score(values, preds2)
        val_acc2 = round(val_acc2 * 100.0, 2)
        print(f'Accuracy model 2: {val_acc2}')
        val_accuracies2.append(val_acc2)

        val_acc3 = accuracy_score(values, preds3)
        val_acc3 = round(val_acc3 * 100.0, 2)
        print(f'Accuracy model 3: {val_acc3}')
        val_accuracies3.append(val_acc3)

    plt.plot(range(3, 6), val_accuracies1, label='Simple model', color=colors[0])
    plt.plot(range(3, 6), val_accuracies2, label='Regular model', color=colors[1])
    plt.plot(range(3, 6), val_accuracies3, label='Complex model', color=colors[2])

    plt.xlabel('Number of input years')
    plt.xticks(np.arange(3, 6, step=1))
    plt.ylabel('Test accuracy')
    plt.ylim(0, 100)
    plt.title('LSTM')
    plt.legend()
    plt.grid()
    plt.show()
    return

def normalization():
    vars = ['tmax', 'vpd', 'def', 'soil', 'ppt', 'PDSI', 'srad']
    regular_df = pd.read_csv('data/12_years_bilinear_interp_w_outliers.csv', on_bad_lines='skip')
    glob_norm_df = regular_df.copy()
    loc_norm_df = regular_df.copy()

    for var in vars:
        var_columns = [col for col in regular_df.columns if re.match(f"^{var}(_year-?\d+_month\d+)$", col)]
        if var_columns:
            values = regular_df[var_columns].values.flatten()
            mean, std = values.mean(), values.std()
            glob_norm_df[var_columns] = (glob_norm_df[var_columns] - mean) / std

    for var in vars:
        var_columns = [col for col in regular_df.columns if re.match(f"^{var}(_year-?\d+_month\d+)$", col)]
        if var_columns:
            row_mean = loc_norm_df[var_columns].mean(axis=1)
            row_std = loc_norm_df[var_columns].std(axis=1).replace(0, 1)
            loc_norm_df[var_columns] = loc_norm_df[var_columns].subtract(row_mean, axis=0).divide(row_std, axis=0)

    val_accuracies_reg = []
    val_accuracies_glob = []
    val_accuracies_loc = []

    for num_years in range(3, 6):
        years = range(1-num_years, 1)
        months = range(1, 13)

        reg_X = np.zeros((len(regular_df), len(vars), len(years) * len(months)))  
        for i, var in enumerate(vars):
            for j, year in enumerate(years):
                for k, month in enumerate(months):
                    col_name = f"{var}_year{year}_month{month}"
                    time_idx = j * len(months) + k
                    reg_X[:, i, time_idx] = regular_df[col_name].values
        reg_X = np.transpose(reg_X, (0, 2, 1))

        glob_X = np.zeros((len(glob_norm_df), len(vars), len(years) * len(months)))  
        for i, var in enumerate(vars):
            for j, year in enumerate(years):
                for k, month in enumerate(months):
                    col_name = f"{var}_year{year}_month{month}"
                    time_idx = j * len(months) + k
                    glob_X[:, i, time_idx] = glob_norm_df[col_name].values
        glob_X = np.transpose(glob_X, (0, 2, 1))

        loc_X = np.zeros((len(loc_norm_df), len(vars), len(years) * len(months)))  
        for i, var in enumerate(vars):
            for j, year in enumerate(years):
                for k, month in enumerate(months):
                    col_name = f"{var}_year{year}_month{month}"
                    time_idx = j * len(months) + k
                    loc_X[:, i, time_idx] = loc_norm_df[col_name].values
        loc_X = np.transpose(loc_X, (0, 2, 1))

        ref_ids = regular_df["Ref_ID"].values
        labels = regular_df["label"].values 

        preds_reg = []
        preds_glob = []
        preds_loc = []
        preds = [preds_reg, preds_glob, preds_loc]
        values = []

        for id in range(0, 153, 3):
            id0 = id
            id1 = id+1
            id2 = id+2
            print(f'Sample: {id0} and {id1} and {id2}. {num_years} years')
            xs = [reg_X, glob_X, loc_X]
            for x in range(0, 3):
                X = xs[x]

                X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []

                # Boolean masks for selecting train/test samples
                train_mask = ~((ref_ids == id0) | (ref_ids == id1)| (ref_ids == id2))
                test_mask = (ref_ids == id0) | (ref_ids == id1) | (ref_ids == id2)

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
                    if x == 0:
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
                    preds[x].extend(y_pred_labels)

        val_acc_reg = accuracy_score(values, preds[0])
        val_acc_reg = round(val_acc_reg * 100.0, 2)
        print(f'Accuracy regular data: {val_acc_reg}')
        val_accuracies_reg.append(val_acc_reg)

        val_acc_glob = accuracy_score(values, preds[1])
        val_acc_glob = round(val_acc_glob * 100.0, 2)
        print(f'Accuracy globally normalized data: {val_acc_glob}')
        val_accuracies_glob.append(val_acc_glob)

        val_acc_loc = accuracy_score(values, preds[2])
        val_acc_loc = round(val_acc_loc * 100.0, 2)
        print(f'Accuracy locally normalized data: {val_acc_loc}')
        val_accuracies_loc.append(val_acc_loc)

    plt.plot(range(3, 6), val_accuracies_reg, label='Unnormalized data', color=colors[0])
    plt.plot(range(3, 6), val_accuracies_glob, label='Globally normalized data', color=colors[1])
    plt.plot(range(3, 6), val_accuracies_loc, label='Locally normalized data', color=colors[2])

    plt.xlabel('Number of input years')
    plt.xticks(np.arange(3, 6, step=1))
    plt.ylabel('Test accuracy')
    plt.ylim(0, 100)
    plt.title('LSTM')
    plt.legend()
    plt.grid()
    plt.show()

def num_input_channels_fine():
    vars8 = ['tmax', 'vpd', 'def', 'srad', 'soil', 'ppt', 'PDSI', 'q']
    vars7 = ['tmax', 'vpd', 'def', 'srad', 'ppt', 'PDSI', 'q']
    vars6 = ['tmax', 'vpd', 'def', 'srad', 'ppt', 'PDSI']
    vars5 = ['tmax', 'def', 'srad', 'ppt', 'PDSI']
    vars4 = ['tmax',  'srad', 'ppt', 'PDSI']
    vars3 = ['srad', 'ppt', 'PDSI']
    vars2 = ['srad', 'PDSI']

    df = pd.read_csv('data/12_years_bilinear_interp_w_outliers.csv', on_bad_lines='skip')

    for n, num_years in enumerate([4, 5, 6]):
        val_accuracies = []
        for vars in [vars2, vars3, vars4, vars5, vars6, vars7, vars8]:
            years = range(1-num_years, 1)
            months = range(1, 13)

            X = np.zeros((len(df), len(vars), len(years) * len(months)))  
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

            for id in range(0, 153, 3):
                id0 = id
                id1 = id+1
                id2 = id+2
                print(f'Sample: {id0} and {id1} and {id2}. {num_years} years and {len(vars)} variables')

                X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []

                # Boolean masks for selecting train/test samples
                train_mask = ~((ref_ids == id0) | (ref_ids == id1)| (ref_ids == id2))
                test_mask = (ref_ids == id0) | (ref_ids == id1) | (ref_ids == id2)

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
            print(f'Accuracy {len(vars)} variables, {num_years} years: {val_acc}')
            val_accuracies.append(val_acc)
        plt.plot(range(2, 9), val_accuracies, label=f'{num_years} years', color=colors[n])

    plt.xlabel('Number of input channels')
    plt.xticks(np.arange(2, 9, step=1))
    plt.ylabel('Test accuracy')
    plt.ylim(0, 100)
    plt.title('LSTM - unnormalized data')
    plt.legend()
    plt.grid()
    plt.show()
    return

if __name__ == "__main__":
    # lstm_vs_gru()
    # monthly_vs_yearly()
    # num_features()
    # num_channels()
    # network_size()
    # normalization()
    num_input_channels_fine()