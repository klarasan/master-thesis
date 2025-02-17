from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from keras import layers

plt.rcParams['font.family'] = 'Times New Roman'

coral = (1,0.5,0.31,0.7)
orange = (1, 0.44, 0, 1)
thistle = (0.9,0.75,0.9,1)
plum = (0.5, 0, 0.5, 1)
slate = (72/255,61/255,139/255, 1)
colors = [coral, orange, thistle, plum, slate]

def init_test():
    vars = ['aet', 'def', 'pet', 'ppt', 'q', 'soil', 'srad', 'swe', 'tmax', 'tmin', 'vap', 'ws', 'vpd', 'PDSI']
    df = pd.read_csv('data/avg_std_12_years_bilinear_interp_w_outliers.csv', on_bad_lines='skip')

    val_accuracies = []
    for num_years in range(1, 13):
        print(f'starting yearly training with {num_years} years')
        times = range(1-num_years, 1)

        cols = ['Ref_ID', 'label']
        # Populate X with values from x
        for _, var in enumerate(vars):
            for _, t in enumerate(times):
                col_name = f"{var}_year{t}_mean"
                cols.append(col_name)
        
        X = df[cols]

        preds = []
        values = []

        for id in range(0, 153):
            x_train = X[X['Ref_ID'] != id]
            y_train = x_train['label']
            x_train = x_train.drop(columns=['label', 'Ref_ID'], axis=1)

            x_test = X[X['Ref_ID'] == id]
            y_test = x_test['label']
            x_test = x_test.drop(columns=['label', 'Ref_ID'], axis=1)

            if y_test.shape[0] > 0:
                model = keras.Sequential([
                layers.Dense(128, activation='relu', input_shape=(x_train.shape[1],)),
                layers.Dropout(0.3),  
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(32, activation='relu'),
                layers.Dense(1, activation='sigmoid') 
                ])
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                _ = model.fit(x_train, y_train, epochs=50, batch_size=32)
                y_pred = model.predict(x_test)
                y_pred_labels = (y_pred > 0.5).astype(int)

                values.extend(y_test)
                preds.extend(y_pred_labels)
                    
        val_acc = accuracy_score(values, preds)
        val_acc = round(val_acc * 100.0, 2)
        print(val_acc)
        val_accuracies.append(val_acc)
    plt.plot(range(1, 13), val_accuracies, label='Yearly data')

    plt.xlabel('Number of input years')
    plt.ylabel('Test accuracy')
    plt.ylim(0, 100)
    plt.title('Multi-layer perceptron')
    plt.legend()
    plt.grid()
    plt.show()
    return

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

            cols = ['Ref_ID', 'label']
            # Populate X with values from x
            for _, var in enumerate(vars):
                for _, t in enumerate(times):
                    col_name = f"{var}_year{t}_mean"
                    cols.append(col_name)
            
            X = df[cols]

            preds = []
            values = []

            for id in range(0, 153):
                x_train = X[X['Ref_ID'] != id]
                y_train = x_train['label']
                x_train = x_train.drop(columns=['label', 'Ref_ID'], axis=1)

                x_test = X[X['Ref_ID'] == id]
                y_test = x_test['label']
                x_test = x_test.drop(columns=['label', 'Ref_ID'], axis=1)

                if y_test.shape[0] > 0:
                    model = keras.Sequential([
                    layers.Dense(128, activation='relu', input_shape=(x_train.shape[1],)),
                    layers.Dropout(0.3),  
                    layers.Dense(64, activation='relu'),
                    layers.Dropout(0.3),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(1, activation='sigmoid') 
                    ])
                    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                    _ = model.fit(x_train, y_train, epochs=50, batch_size=32)
                    y_pred = model.predict(x_test)
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

if __name__ == "__main__":
    # init_test()
    num_features()