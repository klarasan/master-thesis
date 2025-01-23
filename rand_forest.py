import pandas as pd
import train_val_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

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

if __name__ == "__main__":
    train, val, _ = train_val_test_split.datasplit80_10_10()
    x_train, y_train, x_val, y_val = extract_samples(train, val)
    # init_test(x_train, y_train, x_val, y_val)
    m_train, m_val, _ = train_val_test_split.monthly_datasplit70_15_15()
    m_x_train, m_y_train, m_x_val, m_y_val = extract_samples(m_train, m_val)
    monthly_test(m_x_train, m_y_train, m_x_val, m_y_val)