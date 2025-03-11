import re
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.utils import shuffle

plt.rcParams['font.family'] = 'Times New Roman'

coral = (1,0.5,0.31,0.7)
orange = (1, 0.44, 0, 1)
thistle = (0.9,0.75,0.9,1)
plum = (0.5, 0, 0.5, 1)
slate = (72/255,61/255,139/255, 1)
colors = [orange, plum, slate]

def test():
    vars = ['tmax', 'def', 'srad', 'ppt', 'PDSI']
    df = pd.read_csv('data/globally_normalized_12_years_bilinear_interp_w_outliers.csv', on_bad_lines='skip')
    num_years = 5

    df_pos = df[df['label'] == 1]
    df_neg = df[df['label'] == 0]

    var_columns = [col for col in df.columns if re.match("^tmax(_year-?\d+_month\d+)$", col)]
    pos_values = df_pos[var_columns].values.flatten()
    neg_values = df_neg[var_columns].values.flatten()
    pos_mean = pos_values.mean()
    neg_mean = neg_values.mean()

    temp_diff = pos_mean - neg_mean
    df.loc[df['label'] == 0, var_columns] += temp_diff

    accuracies = []
    for end_year in range(0, -11+num_years-2, -1):
        accs = []
        start_year = 1+end_year-num_years
        years = range(start_year, end_year+1)       
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

        for fold, low_id in enumerate(range(1, 153, 19)):
            high_id = low_id + 18
            print(f'Fold: {fold}, low id: {low_id} and high id: {high_id}')
            print(f'Start year: {start_year}, end year: {end_year}')

            preds = []
            values = []

            X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []

            # Boolean masks for selecting train/test samples
            train_mask = (ref_ids < low_id) | (ref_ids > high_id)
            test_mask = (ref_ids >= low_id) & (ref_ids <= high_id)

            X_train_list.append(X[train_mask])
            y_train_list.append(labels[train_mask])

            X_test_list.append(X[test_mask])
            y_test_list.append(labels[test_mask])

            # Convert lists to NumPy arrays
            X_train = np.concatenate(X_train_list, axis=0)
            y_train = np.concatenate(y_train_list, axis=0)

            X_test = np.concatenate(X_test_list, axis=0)
            y_test = np.concatenate(y_test_list, axis=0)

            values = y_test
            model = keras.Sequential([
            layers.GRU(64, return_sequences=True, input_shape=(num_years * 12, len(vars))),
            layers.Dropout(0.2),  
            layers.GRU(32),
            layers.Dropout(0.2),
            layers.Dense(16, activation="relu"), 
            layers.Dense(1, activation="sigmoid") 
            ])
            model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
            X_train, y_train = shuffle(X_train, y_train, random_state=42)
            _ = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
            y_pred = model.predict(X_test)
            y_pred_labels = (y_pred > 0.5).astype(int)
            preds = y_pred_labels

            acc = accuracy_score(values, preds)
            print(f'Fold accuracy: {acc}')
            accs.append(acc)

        print(f'Mean accuracy: {np.mean(accs)}')
        accuracies.append(np.mean(accs)*100.0)

    plt.plot(range(0, -11+num_years-2, -1), accuracies, label=f'{num_years} years', color=plum)

    plt.xlabel('End of data window')
    plt.ylabel('Test accuracy')
    plt.ylim(0, 100)
    plt.title('GRU - adjusted tmax, normalized data')
    plt.legend()
    plt.grid()
    plt.show()

    return

if __name__ == "__main__":
    test()