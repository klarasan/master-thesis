from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from detach_rocket.detach_classes import DetachEnsemble

coral = (1,0.5,0.31,0.7)
orange = (1, 0.44, 0, 1)
thistle = (0.9,0.75,0.9,1)
plum = (0.5, 0, 0.5, 1)
slate = (72/255,61/255,139/255, 1)
colors = [coral, orange, thistle, plum, slate]

def init_test():
    vars = ['tmax', 'vpd', 'def', 'soil', 'ppt', 'PDSI']
    df = pd.read_csv('data/avg_std_12_years_bilinear_interp_w_outliers.csv', on_bad_lines='skip')
    num_years = 9

    times = range(1-num_years, 1)
    X = np.zeros((len(df), len(vars), len(times))) 

    # Populate X with values from x
    for i, var in enumerate(vars):
        for j, t in enumerate(times):
            col_name = f"{var}_year{t}_mean"
            X[:, i, j] = df[col_name].values
    
    ref_ids = df["Ref_ID"].values
    labels = df["label"].values 

    preds = []
    values = []

    for label in reversed(range(0, 2)):
        for id in range(0, 153):
            X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []

            # Boolean masks for selecting train/test samples
            train_mask = (labels != label) | (ref_ids != id)
            test_mask = (labels == label) & (ref_ids == id)

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
                DetachEnsembleModel = DetachEnsemble(num_models=5, num_kernels=1000)
                DetachEnsembleModel.fit(X_train,y_train)
                y_pred = DetachEnsembleModel.predict(X_test)
                preds.extend(y_pred)
                
    val_acc = accuracy_score(values, preds)
    val_acc = round(val_acc * 100.0, 2)
    print(val_acc)
    return

def monthly_vs_yearly_test():
    vars = ['tmax', 'vpd', 'def', 'soil', 'ppt', 'PDSI']
    df = pd.read_csv('data/12_years_bilinear_interp_w_outliers.csv', on_bad_lines='skip')

    val_accuracies = []
    for num_years in range(1, 13):
        print(f'starting monthly training with {num_years} years')
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

        ref_ids = df["Ref_ID"].values
        labels = df["label"].values 

        preds = []
        values = []

        for label in reversed(range(0, 2)):
            for id in range(0, 153):
                X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []

                # Boolean masks for selecting train/test samples
                train_mask = (labels != label) | (ref_ids != id)
                test_mask = (labels == label) & (ref_ids == id)

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
                    DetachEnsembleModel = DetachEnsemble(num_models=5, num_kernels=1000)
                    DetachEnsembleModel.fit(X_train,y_train)
                    y_pred = DetachEnsembleModel.predict(X_test)
                    preds.extend(y_pred)
                    
        val_acc = accuracy_score(values, preds)
        val_acc = round(val_acc * 100.0, 2)
        print(val_acc)
        val_accuracies.append(val_acc)
    print(f'Plotting with {num_years} years')
    plt.plot(range(1, 13), val_accuracies, label='Monthly data', color=colors[3])

    df = pd.read_csv('data/avg_std_12_years_bilinear_interp_w_outliers.csv', on_bad_lines='skip')

    val_accuracies = []
    for num_years in range(9, 13):
        print(f'starting yearly training with {num_years} years')
        times = range(1-num_years, 1)
        X = np.zeros((len(df), len(vars), len(times))) 

        # Populate X with values from x
        for i, var in enumerate(vars):
            for j, t in enumerate(times):
                col_name = f"{var}_year{t}_mean"
                X[:, i, j] = df[col_name].values
        
        ref_ids = df["Ref_ID"].values
        labels = df["label"].values 

        preds = []
        values = []

        for label in reversed(range(0, 2)):
            for id in range(0, 153):
                X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []

                # Boolean masks for selecting train/test samples
                train_mask = (labels != label) | (ref_ids != id)
                test_mask = (labels == label) & (ref_ids == id)

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
                    DetachEnsembleModel = DetachEnsemble(num_models=5, num_kernels=1000)
                    DetachEnsembleModel.fit(X_train,y_train)
                    y_pred = DetachEnsembleModel.predict(X_test)
                    preds.extend(y_pred)
                    
        val_acc = accuracy_score(values, preds)
        val_acc = round(val_acc * 100.0, 2)
        print(val_acc)
        val_accuracies.append(val_acc)
    print(f'Plotting with {num_years} years')
    plt.plot(range(9, 13), val_accuracies, label='Yearly data', color=colors[1])

    plt.xlabel('Number of input years')
    plt.ylabel('Test accuracy')
    plt.ylim(0, 100)
    plt.title('Detach Rocket')
    plt.legend()
    plt.grid()
    plt.show()
    return

def feat_imp():
    vars14 = ['aet', 'def', 'pet', 'ppt', 'q', 'soil', 'srad', 'swe', 'tmax', 'tmin', 'vap', 'ws', 'vpd', 'PDSI']
    vars10 = ['aet', 'def', 'ppt', 'q', 'soil', 'srad', 'tmax', 'vap', 'vpd', 'PDSI']
    vars8 = ['tmax', 'vpd', 'def', 'soil', 'ppt', 'PDSI', 'srad', 'q']
    vars6 = ['tmax', 'vpd', 'def', 'soil', 'ppt', 'PDSI']
    vars3 = ['srad', 'def', 'PDSI']
    vars_list = [vars14, vars10, vars8, vars6, vars3]

    df = pd.read_csv('data/12_years_bilinear_interp_w_outliers.csv', on_bad_lines='skip')

    for i in range(0, 5):
        val_accuracies = []
        vars = vars_list[i]
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

            ref_ids = df["Ref_ID"].values
            labels = df["label"].values 

            preds = []
            values = []

            for label in reversed(range(0, 2)):
                for id in range(0, 153):
                    X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []

                    # Boolean masks for selecting train/test samples
                    train_mask = (labels != label) | (ref_ids != id)
                    test_mask = (labels == label) & (ref_ids == id)

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
                        print(f'Sample: {id}, label {label}. {num_years} years and {len(vars)} variables')
                        values.extend(y_test)
                        DetachEnsembleModel = DetachEnsemble(num_models=5, num_kernels=1000)
                        DetachEnsembleModel.fit(X_train,y_train)
                        y_pred = DetachEnsembleModel.predict(X_test)
                        preds.extend(y_pred)
                        
            val_acc = accuracy_score(values, preds)
            val_acc = round(val_acc * 100.0, 2)
            print(f'Accuracy: {val_acc}%')
            val_accuracies.append(val_acc)
        plt.plot(range(2, 13, 2), val_accuracies, label=f'{len(vars)} input channels', color=colors[i])
    plt.xlabel('Number of input years')
    plt.ylabel('Test accuracy')
    plt.ylim(0, 100)
    plt.title('Detach Rocket')
    plt.legend()
    plt.grid()
    plt.show()
    return

if __name__ == "__main__":
    # init_test()
    # monthly_vs_yearly_test()
    feat_imp()