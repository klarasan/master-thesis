from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from detach_rocket.detach_classes import DetachEnsemble
import re

plt.rcParams['font.family'] = 'Times New Roman'

coral = (1,0.5,0.31,0.7)
orange = (1, 0.44, 0, 1)
thistle = (0.9,0.75,0.9,1)
plum = (0.5, 0, 0.5, 1)
slate = (72/255,61/255,139/255, 1)
colors = [orange, plum, slate]

def test():
    vars = ['tmax', 'vpd', 'def', 'srad', 'ppt', 'PDSI']
    df = pd.read_csv('data/12_years_bilinear_interp_w_outliers.csv', on_bad_lines='skip')
    num_years = 6
    
    for n, num_years in enumerate(range(5, 7)):
        val_accuracies = []
        for end_year in range(0, -11+num_years-2, -1):
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
            
            ref_ids = df["Ref_ID"].values
            labels = df["label"].values 

            preds = []
            values = []

            for id in range(0, 153, 3):
                id0 = id
                id1 = id+1
                id2 = id+2
                print(f'Sample: {id0} and {id1} and {id2}. {len(years)} years. \nStart year: {start_year}, end year: {end_year}')

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
                    DetachEnsembleModel = DetachEnsemble(num_models=1, trade_off=0.2)
                    DetachEnsembleModel.fit(X_train, y_train)
                    y_pred = DetachEnsembleModel.predict(X_test)
                    preds.extend(y_pred)

            val_acc = accuracy_score(values, preds)
            val_acc = round(val_acc * 100.0, 2)
            print(f'Accuracy {num_years} years: {val_acc}')
            val_accuracies.append(val_acc)

        plt.plot(range(0, -11+num_years-2,-1), val_accuracies, label=f'{num_years} years', color=colors[n])

    plt.xlabel('End of data window')
    plt.ylabel('Test accuracy')
    plt.ylim(0, 100)
    plt.title('Detach Rocket')
    plt.legend()
    plt.grid()
    plt.show()

    return

if __name__ == "__main__":
    test()