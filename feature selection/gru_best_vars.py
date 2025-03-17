from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow import keras
from keras import layers

def test():
    best_vars = ['PDSI', 'srad', 'ppt', 'tmax', 'def']
    mid_vars = ['vpd', 'vap', 'pet', 'aet']
    worst_vars = ['swe', 'ws', 'tmin', 'soil', 'q']
    var_subsets = [best_vars, mid_vars, worst_vars]

    years = [5, 12]

    for num_years in years:
        for v, vars in enumerate(var_subsets):
            df = pd.read_csv('data/globally_normalized_12_years_bilinear_interp_w_outliers.csv', on_bad_lines='skip')

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

            accuracies = np.zeros(8)

            for fold, low_id in enumerate(range(1, 153, 19)):
                high_id = low_id + 18
                print(f'Fold: {fold}, low id: {low_id} and high id: {high_id}')

                X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []

                train_mask = (ref_ids < low_id) | (ref_ids > high_id)
                test_mask = (ref_ids >= low_id) & (ref_ids <= high_id)

                X_train_list.append(X[train_mask])
                y_train_list.append(labels[train_mask])

                X_test_list.append(X[test_mask])
                y_test_list.append(labels[test_mask])

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
                print(f'\nAccuracy: {acc}\n')

                accuracies[fold] = acc
            if v == 0:
                level = 'best'
            elif v == 1:
                level = 'mid'
            else: 
                level = 'worst'
            results = pd.DataFrame([accuracies], columns=['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'Fold 6', 'Fold 7', 'Fold 8'])
            results.to_csv(f'gru_vars_comp_{level}_{num_years}years_normalized.csv', index=False)
    return

if __name__ == "__main__":
    test()