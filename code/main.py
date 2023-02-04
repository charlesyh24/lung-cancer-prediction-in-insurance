import pandas as pd
import preprocessing as pre
import os

path = os.path.join('..', 'data', 'lung-cancer-patient.csv')
df = pd.read_csv(path)
X_train, X_test, y_train, y_test, scaler, pca = pre.preprocess_data(df)