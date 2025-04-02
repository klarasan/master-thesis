import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle

vars = ['PDSI']

df = pd.read_csv('data/avg_std_12_years_bilinear_interp_w_outliers.csv', on_bad_lines='skip')
num_years = 12

cols = []
for var in vars:
    for year in range(1-num_years, 1):
        cols.append(f'{var}_year{year}_mean')
curr_df = df[cols]

ref_ids = df["Ref_ID"].values
labels = df["label"].values 

accuracies = np.zeros((3,8))
for run in range(0, 3):
    for fold, low_id in enumerate(range(1, 153, 19)):
        high_id = low_id + 18
        print(f'Run: {run}, fold: {fold}, low id: {low_id} and high id: {high_id}')

        preds = []
        values = []

        X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []

        train_mask = (ref_ids < low_id) | (ref_ids > high_id)
        test_mask = (ref_ids >= low_id) & (ref_ids <= high_id)

        X_train_list.append(curr_df[train_mask])
        y_train_list.append(labels[train_mask])

        X_test_list.append(curr_df[test_mask])
        y_test_list.append(labels[test_mask])

        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)

        X_test = np.concatenate(X_test_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)

        values = y_test
        X_train, y_train = shuffle(X_train, y_train, random_state=42)
        rf_model = RandomForestClassifier(n_estimators=300, random_state=42)
        rf_model.fit(X_train, y_train)
        preds = rf_model.predict(X_test)

        acc = accuracy_score(values, preds)
        print(f'\nAccuracy: {acc}\n')
        accuracies[run][fold] = acc

results = pd.DataFrame(accuracies, columns=['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'Fold 6', 'Fold 7', 'Fold 8'])
results.to_csv('rf_model_comparison.csv', index=False)
