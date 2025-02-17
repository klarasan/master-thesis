import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

plt.rcParams['font.family'] = 'Times New Roman'

coral = (1,0.5,0.31,0.7)
orange = (1, 0.44, 0, 1)
thistle = (0.9,0.75,0.9,1)
plum = (0.5, 0, 0.5, 1)
slate = (72/255,61/255,139/255, 1)
colors = [coral, orange, thistle, plum, slate]

VARS = ['aet', 'def', 'pet', 'ppt', 'q', 'soil', 'srad', 'swe', 'tmax', 'tmin', 'vap', 'ws', 'vpd', 'PDSI']

def extract_samples(train, val):
    x_train = train.drop(columns=['label'])
    y_train = train['label']
    x_val= val.drop(columns=['label'])
    y_val= val['label']

    return x_train, y_train, x_val, y_val

def monthly_test(X_train, Y_train, X_val, Y_val):
    for n in range(3, 18, 3):
        val_accuracies = []
        for i in reversed(range(-11, 1)):
            dropped_columns = ['Ref_ID', 'Year', 'Latitude', 'Longitude'] 
            for var in VARS:
                for year_offset in range(-11, i):
                    for m in range(1,13):
                        dropped_columns.append(f'{var}_year{year_offset}_month{m}')            
            x_train = X_train.drop(columns=dropped_columns, axis=1)
            x_val = X_val.drop(columns=dropped_columns, axis=1)

            x_train, y_train = shuffle(x_train, Y_train, random_state=42)

            rf_model = RandomForestClassifier(n_estimators=n, random_state=42)

            rf_model.fit(x_train, y_train)
            
            val_pred = rf_model.predict(x_val)
            val_acc = accuracy_score(Y_val, val_pred)
            val_acc = round(val_acc * 100.0, 2)
            val_accuracies.append(val_acc)
        print(f'Plotting with n: {n}')
        plt.plot(range(1, 13), val_accuracies, label=f'{n} classifiers', color=colors[int(n/3)-1])

    plt.xlabel('Number of input years')
    plt.ylabel('Validation accuracy')
    plt.ylim(0, 100)
    plt.title('Random forest 70-15-15 split (monthly)')
    plt.legend()
    plt.grid()
    plt.show()
    return

def init_test(X_train, Y_train, X_val, Y_val):
    for n in range(3, 18, 3):
        val_accuracies = []
        for i in reversed(range(-11, 1)):
            dropped_columns = ['Ref_ID', 'Year', 'Latitude', 'Longitude'] 
            for var in VARS:
                for year in range(-11, 1):
                    dropped_columns.append(f'{var}_year{year}_std')
                for year_offset in range(-11, i):
                    dropped_columns.append(f'{var}_year{year_offset}_mean')            
            x_train = X_train.drop(columns=dropped_columns, axis=1)
            x_val = X_val.drop(columns=dropped_columns, axis=1)

            x_train, y_train = shuffle(x_train, Y_train, random_state=42)

            rf_model = RandomForestClassifier(n_estimators=n, random_state=42)

            rf_model.fit(x_train, y_train)
            
            val_pred = rf_model.predict(x_val)
            val_acc = accuracy_score(Y_val, val_pred)
            val_acc = round(val_acc * 100.0, 2)

            val_accuracies.append(val_acc)

        plt.plot(range(1, 13), val_accuracies, label=f'{n} classifiers', color=colors[int(n/3)-1])

    plt.xlabel('Number of input years')
    plt.ylabel('Validation accuracy')
    plt.ylim(0, 100)
    plt.title('Random forest 80-10-10 datasplit')
    plt.legend()
    plt.grid()
    plt.show()
    return

def drop_columns(start, end):
    columns_to_drop = ['Year', 'Latitude', 'Longitude']
    for var in VARS:
        for year in range(-11, 1):
            columns_to_drop.append(f'{var}_year{year}_std')
        for year_offset in range(start, end):
            columns_to_drop.append(f'{var}_year{year_offset}_mean')
    return columns_to_drop

def leave_one_out():
    all_data_df = pd.read_csv('data/avg_std_12_years_bilinear_interp_w_outliers.csv', on_bad_lines='skip')
    i = 0

    for n_classifiers in range(3, 18, 3):
        val_accuracies = []
        for num_years in range(2, 13, 2):
            dropped_columns = drop_columns(-11, 1-num_years)
            df = all_data_df.drop(columns=dropped_columns, axis=1)

            test_pred = []
            test_value = []

            for label in reversed(range(0,2)):
                for id in range(0, 153):
                    x_train = df[(df['label'] != label) | (df['Ref_ID'] != id)]
                    y_train = x_train['label']
                    x_train = x_train.drop(columns=['label', 'Ref_ID'], axis=1)
    
                    x_test = df[(df['label'] == label) & (df['Ref_ID'] == id)]
                    y_test = x_test['label']
                    x_test = x_test.drop(columns=['label', 'Ref_ID'], axis=1)

                    if y_test.shape[0] != 0:
                        test_value.extend(y_test.values)
                        x_train, y_train = shuffle(x_train, y_train, random_state=42)
                        rf_model = RandomForestClassifier(n_estimators=n_classifiers, random_state=42)
                        rf_model.fit(x_train, y_train)
                        
                        val_pred = rf_model.predict(x_test)
                        test_pred.extend(val_pred)

            val_acc = accuracy_score(test_value, test_pred)
            val_acc = round(val_acc * 100.0, 2)
            print(val_acc)
            val_accuracies.append(val_acc)
        print(f'Finished training with {n_classifiers} classifiers')
        plt.plot(range(2, 13, 2), val_accuracies, label=f'{n_classifiers} classifiers', color=colors[i])
        i += 1
    
    plt.xlabel('Number of input years')
    plt.ylabel('Test accuracy')
    plt.ylim(0, 100)
    plt.title('Random forest leave one out')
    plt.legend()
    plt.grid()
    plt.show()

    return

def kfold_test():
    df = pd.read_csv('data/avg_std_12_years_closest_gridpoint.csv', on_bad_lines='skip')
    cols = []
    for var in VARS:
        for year in range(-3, 1):
            cols.append(f'{var}_year{year}_mean')
    x = df[cols]
    y = df['label']
    
    classifier = RandomForestClassifier(n_estimators=800, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(classifier, x, y, cv=kf, scoring='accuracy')
    
    mean_cv_score = np.mean(cv_scores)
    std_cv_score = np.std(cv_scores)
    
    print(f"K-Fold Cross Validation (k={5}):")
    print(f"Accuracy per fold: {cv_scores}")
    print(f"Mean Accuracy: {mean_cv_score:.4f}")
    print(f"Standard Deviation: {std_cv_score:.4f}")

def feat_imp():
    # vars14 = ['aet', 'def', 'pet', 'ppt', 'q', 'soil', 'srad', 'swe', 'tmax', 'tmin', 'vap', 'ws', 'vpd', 'PDSI']
    # vars10 = ['aet', 'def', 'ppt', 'q', 'soil', 'srad', 'tmax', 'vap', 'vpd', 'PDSI']
    # vars8 = ['tmax', 'vpd', 'def', 'soil', 'ppt', 'PDSI', 'srad', 'q']
    # vars6 = ['tmax', 'vpd', 'def', 'soil', 'ppt', 'PDSI']
    # vars3 = ['srad', 'def', 'PDSI']
    # vars1 = ['PDSI']
    # vars2 = ['PDSI', 'def']
    # vars3 = ['PDSI', 'def', 'srad']
    # vars5 = ['PDSI', 'def', 'srad', 'vpd', 'soil']
    # vars7 = ['PDSI', 'def', 'srad', 'vpd', 'soil', 'tmax', 'ppt']
    labels = ['ppt + srad', 'ppt + tmax', 'def + srad', 'def + tmax', 'PDSI']
    vars1 = ['ppt', 'srad']
    vars2 = ['ppt', 'tmax']
    vars3 = ['def', 'srad']
    vars4 = ['def', 'tmax']
    vars5 = ['PDSI']

    vars = [vars1, vars2, vars3, vars4, vars5]
    df = pd.read_csv('data/avg_std_12_years_bilinear_interp_w_outliers.csv', on_bad_lines='skip')

    for i in range(0, 5):
        val_accuracies = []
        for num_years in range(1, 13):
            cols = ['Ref_ID', 'label']
            for var in vars[i]:
                for year in range(1-num_years, 1):
                    cols.append(f'{var}_year{year}_mean')
            x = df[cols]

            preds = []
            values = []

            for label in reversed(range(0,2)):
                for id in range(0, 153):
                    x_train = x[(x['label'] != label) | (x['Ref_ID'] != id)]
                    y_train = x_train['label']
                    x_train = x_train.drop(columns=['label', 'Ref_ID'], axis=1)

                    x_test = x[(x['label'] == label) & (x['Ref_ID'] == id)]
                    y_test = x_test['label']
                    x_test = x_test.drop(columns=['label', 'Ref_ID'], axis=1)

                    if y_test.shape[0] != 0:
                        values.extend(y_test.values)
                        x_train, y_train = shuffle(x_train, y_train, random_state=42)
                        rf_model = RandomForestClassifier(n_estimators=12, random_state=42)
                        rf_model.fit(x_train, y_train)
                        
                        pred = rf_model.predict(x_test)
                        preds.extend(pred)
    
            val_acc = accuracy_score(values, preds)
            val_acc = round(val_acc * 100.0, 2)
            print(val_acc)
            val_accuracies.append(val_acc)
        print(f'Finished training round {i+1}')
        plt.plot(range(1, 13), val_accuracies, label=labels[i], color=colors[i])
    
    plt.xlabel('Number of input years')
    plt.ylabel('Test accuracy')
    plt.ylim(0, 100)
    plt.title('Random forest leave one out')
    plt.legend()
    plt.grid()
    plt.show()
    return

def dataset_test():
    df_closest = pd.read_csv('data/avg_std_12_years_closest_gridpoint.csv', on_bad_lines='skip')
    df_interp = pd.read_csv('data/avg_std_12_years_bilinear_interp_w_outliers.csv', on_bad_lines='skip')
    dfs = [df_closest, df_interp]
    vars = ['tmax', 'vpd', 'def', 'soil', 'ppt', 'PDSI']

    for i in range(0, 2):
        val_accuracies = []
        for num_years in range(1, 13):
            cols = ['Ref_ID', 'label']
            for var in vars:
                for year in range(1-num_years, 1):
                    cols.append(f'{var}_year{year}_mean')
            x = dfs[i][cols]

            preds = []
            values = []

            for label in reversed(range(0,2)):
                for id in range(0, 153):
                    x_train = x[(x['label'] != label) | (x['Ref_ID'] != id)]
                    y_train = x_train['label']
                    x_train = x_train.drop(columns=['label', 'Ref_ID'], axis=1)

                    x_test = x[(x['label'] == label) & (x['Ref_ID'] == id)]
                    y_test = x_test['label']
                    x_test = x_test.drop(columns=['label', 'Ref_ID'], axis=1)

                    if y_test.shape[0] != 0:
                        values.extend(y_test.values)
                        x_train, y_train = shuffle(x_train, y_train, random_state=42)
                        rf_model = RandomForestClassifier(n_estimators=12, random_state=42)
                        rf_model.fit(x_train, y_train)
                        
                        pred = rf_model.predict(x_test)
                        preds.extend(pred)
    
            val_acc = accuracy_score(values, preds)
            val_acc = round(val_acc * 100.0, 2)
            print(val_acc)
            val_accuracies.append(val_acc)
        print(f'Finished training round {i+1}')
        label = 'Closest grid value' if i == 0 else 'Interpolated grid value'
        plt.plot(range(1, 13), val_accuracies, label=label, color=colors[i])
    
    plt.xlabel('Number of input years')
    plt.ylabel('Test accuracy')
    plt.ylim(0, 100)
    plt.title('Random forest leave one out')
    plt.legend()
    plt.grid()
    plt.show()
    return
    
if __name__ == "__main__":
    # train, val, _ = train_val_test_split.datasplit80_10_10()
    # x_train, y_train, x_val, y_val = extract_samples(train, val)
    # init_test(x_train, y_train, x_val, y_val)
    # m_train, m_val, _ = train_val_test_split.monthly_datasplit70_15_15()
    # m_x_train, m_y_train, m_x_val, m_y_val = extract_samples(m_train, m_val)
    # monthly_test(m_x_train, m_y_train, m_x_val, m_y_val)
    # leave_one_out()
    # kfold_test()
    feat_imp()
    # dataset_test()