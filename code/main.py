import pandas as pd
import preprocessing as pre
import modeling as mdl
import application as appl
import os
import warnings
warnings.filterwarnings('ignore')

path = os.path.join('..', 'data', 'lung-cancer-patient.csv')
df = pd.read_csv(path)

X_train, X_test, y_train, y_test, scaler, pca = pre.preprocess_data(df)

model = mdl.data_modeling(X_train, X_test, y_train, y_test)

appl.insurance_appl(scaler, pca, model)