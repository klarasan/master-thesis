from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from detach_rocket.detach_classes import DetachEnsemble
from matplotlib.colors import LinearSegmentedColormap
import re

plt.rcParams['font.family'] = 'Times New Roman'

coral = (1,0.5,0.31,0.7)
orange = (1, 0.44, 0, 1)
thistle = (0.9,0.75,0.9,1)
plum = (0.5, 0, 0.5, 1)
slate = (72/255,61/255,139/255, 1)
colors = [coral, orange, thistle, plum, slate]
#colors = [orange, plum, slate]

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

def num_input_channels():
    vars14 = ['aet', 'def', 'pet', 'ppt', 'q', 'soil', 'srad', 'swe', 'tmax', 'tmin', 'vap', 'ws', 'vpd', 'PDSI']
    vars10 = ['aet', 'def', 'ppt', 'q', 'soil', 'srad', 'tmax', 'vap', 'vpd', 'PDSI']
    vars8 = ['tmax', 'vpd', 'def', 'soil', 'ppt', 'PDSI', 'srad', 'q']
    vars6 = ['tmax', 'vpd', 'def', 'soil', 'ppt', 'PDSI']
    vars3 = ['srad', 'def', 'PDSI']
    vars_list = [vars14, vars10, vars8, vars6, vars3]

    df = pd.read_csv('data/12_years_bilinear_interp_w_outliers.csv', on_bad_lines='skip')
    
    for v in range(0, 5):
        val_accuracies = []
        variables = vars_list[v]
        for num_years in range(2, 13, 2):
            years = range(1-num_years, 1)
            months = range(1, 13)
            X = np.zeros((len(df), len(variables), len(years) * len(months)))  

            # Populate X with values from df
            for i, var in enumerate(variables):
                for j, year in enumerate(years):
                    for k, month in enumerate(months):
                        col_name = f"{var}_year{year}_month{month}"
                        time_idx = j * len(months) + k
                        X[:, i, time_idx] = df[col_name].values

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
                    print(f'Sample: {id}. {num_years} years and {len(variables)} variables')
                    values.extend(y_test)
                    DetachEnsembleModel = DetachEnsemble(num_models=5, num_kernels=1000)
                    DetachEnsembleModel.fit(X_train,y_train)
                    y_pred = DetachEnsembleModel.predict(X_test)
                    preds.extend(y_pred)
                        
            val_acc = accuracy_score(values, preds)
            val_acc = round(val_acc * 100.0, 2)
            print(f'Accuracy: {val_acc}%')
            val_accuracies.append(val_acc)
        plt.plot(range(2, 13, 2), val_accuracies, label=f'{len(variables)} input channels', color=colors[v])
    plt.xlabel('Number of input years')
    plt.ylabel('Test accuracy')
    plt.ylim(0, 100)
    plt.title('Detach Rocket')
    plt.legend()
    plt.grid()
    plt.show()
    return

def channel_imp():
    vars14 = ['aet', 'def', 'pet', 'ppt', 'q', 'soil', 'srad', 'swe', 'tmax', 'tmin', 'vap', 'ws', 'vpd', 'PDSI']
    # vars10 = ['aet', 'def', 'ppt', 'pet', 'soil', 'srad', 'tmax', 'vap', 'vpd', 'PDSI']
    # vars8 = ['tmax', 'vpd', 'def', 'soil', 'ppt', 'PDSI', 'srad', 'q']
    # vars6 = ['tmax', 'vpd', 'def', 'srad', 'ppt', 'PDSI']
    # vars3 = ['srad', 'ppt', 'PDSI']

    colors = ["orange", "coral", "thistle", "plum"]
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=100)
   
    num_years = 4
    years = range(1-num_years, 1)
    months = range(1, 13)

    feature_importance = pd.DataFrame()
    feature_importance['variable'] = vars14
    feature_importance['importance'] = np.zeros(14)

    df = pd.read_csv('data/12_years_bilinear_interp_w_outliers.csv', on_bad_lines='skip')
    
    X = np.zeros((len(df), len(vars14), len(years) * len(months)))  

    # Populate X with values from df
    for i, var in enumerate(vars14):
        for j, year in enumerate(years):
            for k, month in enumerate(months):
                col_name = f"{var}_year{year}_month{month}"
                time_idx = j * len(months) + k
                X[:, i, time_idx] = df[col_name].values

    labels = df["label"].values 
    for _ in range(0, 15):
        DetachEnsembleModel = DetachEnsemble(num_models=25)
        DetachEnsembleModel.fit(X, labels)

        channel_relevance = DetachEnsembleModel.estimate_channel_relevance()
        feature_importance['importance'] += channel_relevance
    feature_importance = feature_importance.sort_values(by='importance', ascending=False)

    norm = np.linspace(0, 1, len(vars14))
    bar_colors = custom_cmap(norm)

    plt.figure(figsize=(8,3.5))
    plt.bar(feature_importance['variable'], feature_importance['importance'], color=bar_colors, zorder=2)
    plt.title(f'Channel relevance estimation, {len(vars14)} channels')
    plt.grid(True, linestyle='-', alpha=0.5, zorder=1)
    plt.xlabel('Input channels')
    plt.ylabel('Relevance Estimation (arb. unit)')
    plt.tight_layout()
    plt.show()
    return

def hyper_param():
    vars = ['aet', 'def', 'pet', 'ppt', 'srad', 'tmax', 'tmin', 'vap', 'vpd', 'PDSI']
    df = pd.read_csv('data/12_years_bilinear_interp_w_outliers.csv', on_bad_lines='skip')

    colors = ["orange", "coral", "thistle", "plum"]
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=100)
    norm = np.linspace(0, 1, 5)
    line_colors = custom_cmap(norm)
    c = 0

    for num_kernels in [100, 500, 1000, 2000, 5000]:
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

                X_train = np.concatenate(X_train_list, axis=0)
                y_train = np.concatenate(y_train_list, axis=0)

                X_test = np.concatenate(X_test_list, axis=0)
                y_test = np.concatenate(y_test_list, axis=0)

                if y_test.shape[0] > 0:
                    print(f'Sample: {id}. {num_years} years and {num_kernels} kernels')
                    values.extend(y_test)
                    DetachEnsembleModel = DetachEnsemble(num_models=5, num_kernels=num_kernels)
                    DetachEnsembleModel.fit(X_train,y_train)
                    y_pred = DetachEnsembleModel.predict(X_test)
                    preds.extend(y_pred)
                        
            val_acc = accuracy_score(values, preds)
            val_acc = round(val_acc * 100.0, 2)
            print(f'Accuracy: {val_acc}%')
            val_accuracies.append(val_acc)
        plt.plot(range(2, 13, 2), val_accuracies, label=f'{num_kernels} kernels', color=line_colors[c])
        c+=1
    plt.xlabel('Number of input years')
    plt.ylabel('Test accuracy')
    plt.ylim(0, 100)
    plt.title('Detach Rocket')
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

    for num_years in range(2, 11):
        years = range(1-num_years, 1)
        months = range(1, 13)

        reg_X = np.zeros((len(regular_df), len(vars), len(years) * len(months)))  
        for i, var in enumerate(vars):
            for j, year in enumerate(years):
                for k, month in enumerate(months):
                    col_name = f"{var}_year{year}_month{month}"
                    time_idx = j * len(months) + k
                    reg_X[:, i, time_idx] = regular_df[col_name].values

        glob_X = np.zeros((len(glob_norm_df), len(vars), len(years) * len(months)))  
        for i, var in enumerate(vars):
            for j, year in enumerate(years):
                for k, month in enumerate(months):
                    col_name = f"{var}_year{year}_month{month}"
                    time_idx = j * len(months) + k
                    glob_X[:, i, time_idx] = glob_norm_df[col_name].values

        loc_X = np.zeros((len(loc_norm_df), len(vars), len(years) * len(months)))  
        for i, var in enumerate(vars):
            for j, year in enumerate(years):
                for k, month in enumerate(months):
                    col_name = f"{var}_year{year}_month{month}"
                    time_idx = j * len(months) + k
                    loc_X[:, i, time_idx] = loc_norm_df[col_name].values

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
                    DetachEnsembleModel = DetachEnsemble(num_models=1, num_kernels=2000)
                    DetachEnsembleModel.fit(X_train, y_train)
                    y_pred = DetachEnsembleModel.predict(X_test)
                    preds[x].extend(y_pred)

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

    plt.plot(range(2, 11), val_accuracies_reg, label='Unnormalized data', color=colors[0])
    plt.plot(range(2, 11), val_accuracies_glob, label='Globally normalized data', color=colors[1])
    plt.plot(range(2, 11), val_accuracies_loc, label='Locally normalized data', color=colors[2])

    plt.xlabel('Number of input years')
    plt.xticks(np.arange(2, 11, step=1))
    plt.ylabel('Test accuracy')
    plt.ylim(0, 100)
    plt.title('Detach Rocket')
    plt.legend()
    plt.grid()
    plt.show()
    return

def pruning():
    vars = ['tmax', 'vpd', 'def', 'soil', 'ppt', 'PDSI', 'srad']
    df = pd.read_csv('data/12_years_bilinear_interp_w_outliers.csv', on_bad_lines='skip')

    val_accuracies002 = []
    val_accuracies005 = []
    val_accuracies01 = []
    val_accuracies015 = []
    val_accuracies02 = []
    for num_years in range(3, 8):
        years = range(1-num_years, 1)
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

        preds002 = []
        preds005 = []
        preds01 = []
        preds015 = []
        preds02 = []
        values = []

        for id in range(0, 153, 3):
            id0 = id
            id1 = id+1
            id2 = id+2
            print(f'Sample: {id0} and {id1} and {id2}. {num_years} years')

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
                DetachEnsembleModel = DetachEnsemble(num_models=1, trade_off=0.02)
                DetachEnsembleModel.fit(X_train, y_train)
                y_pred = DetachEnsembleModel.predict(X_test)
                preds002.extend(y_pred)

                DetachEnsembleModel = DetachEnsemble(num_models=1, trade_off=0.05)
                DetachEnsembleModel.fit(X_train, y_train)
                y_pred = DetachEnsembleModel.predict(X_test)
                preds005.extend(y_pred)

                DetachEnsembleModel = DetachEnsemble(num_models=1, trade_off=0.1)
                DetachEnsembleModel.fit(X_train, y_train)
                y_pred = DetachEnsembleModel.predict(X_test)
                preds01.extend(y_pred)

                DetachEnsembleModel = DetachEnsemble(num_models=1, trade_off=0.15)
                DetachEnsembleModel.fit(X_train, y_train)
                y_pred = DetachEnsembleModel.predict(X_test)
                preds015.extend(y_pred)

                DetachEnsembleModel = DetachEnsemble(num_models=1, trade_off=0.2)
                DetachEnsembleModel.fit(X_train, y_train)
                y_pred = DetachEnsembleModel.predict(X_test)
                preds02.extend(y_pred)
        
        val_acc = accuracy_score(values, preds002)
        val_acc = round(val_acc * 100.0, 2)
        print(f'Accuracy 0.02 trade-off: {val_acc}')
        val_accuracies002.append(val_acc)

        val_acc = accuracy_score(values, preds005)
        val_acc = round(val_acc * 100.0, 2)
        print(f'Accuracy 0.05 trade-off: {val_acc}')
        val_accuracies005.append(val_acc)

        val_acc = accuracy_score(values, preds01)
        val_acc = round(val_acc * 100.0, 2)
        print(f'Accuracy 0.1 trade-off: {val_acc}')
        val_accuracies01.append(val_acc)

        val_acc = accuracy_score(values, preds015)
        val_acc = round(val_acc * 100.0, 2)
        print(f'Accuracy 0.15 trade-off: {val_acc}')
        val_accuracies015.append(val_acc)

        val_acc = accuracy_score(values, preds02)
        val_acc = round(val_acc * 100.0, 2)
        print(f'Accuracy 0.2 trade-off: {val_acc}')
        val_accuracies02.append(val_acc)

    plt.plot(range(3, 8), val_accuracies002, label='Trade-off: 0.02', color=colors[0])
    plt.plot(range(3, 8), val_accuracies005, label='Trade-off: 0.05', color=colors[1])
    plt.plot(range(3, 8), val_accuracies01, label='Trade-off: 0.1', color=colors[2])
    plt.plot(range(3, 8), val_accuracies015, label='Trade-off: 0.15', color=colors[3])
    plt.plot(range(3, 8), val_accuracies02, label='Trade-off: 0.2', color=colors[4])

    plt.xlabel('Number of input years')
    plt.xticks(np.arange(3, 8, step=1))
    plt.ylabel('Test accuracy')
    plt.ylim(0, 100)
    plt.title('Detach Rocket')
    plt.legend()
    plt.grid()
    plt.show()
    return

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
                    DetachEnsembleModel = DetachEnsemble(num_models=1, trade_off=0.02)
                    DetachEnsembleModel.fit(X_train, y_train)
                    y_pred = DetachEnsembleModel.predict(X_test)
                    preds.extend(y_pred)
            
            val_acc = accuracy_score(values, preds)
            val_acc = round(val_acc * 100.0, 2)
            print(f'Accuracy {len(vars)} variables, {num_years} years: {val_acc}')
            val_accuracies.append(val_acc)
            
        plt.plot(range(2, 9), val_accuracies, label=f'{num_years} years', color=colors[n])

    plt.xlabel('Number of input channels')
    plt.xticks(np.arange(2, 9, step=1))
    plt.ylabel('Test accuracy')
    plt.ylim(0, 100)
    plt.title('Detach Rocket - unnormalized data')
    plt.legend()
    plt.grid()
    plt.show()
    return

if __name__ == "__main__":
    # init_test()
    # monthly_vs_yearly_test()
    # num_input_channels()
    channel_imp()
    # hyper_param()
    # normalization()
    # pruning()
    #num_input_channels_fine()