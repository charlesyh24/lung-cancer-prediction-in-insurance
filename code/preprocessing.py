import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def drop_dupl(df):
    """drop duplicates"""
    df = df.drop_duplicates().reset_index(drop=True)
    
    return df

def uni_cols(df):
    """uniform column names"""
    list_col = []

    for col in df.columns:
        if col[-1] == ' ':
            list_col.append(col[:-1])
        else:
            list_col.append(col.replace(' ', '_'))

    df.columns = list_col
    
    return df

def cat_cols(df):
    """convert into categories and binary numerical values"""
    dict_repl = {'M': 1, 'F': 0, 'YES': 1, 'NO': 0, 2: 1, 1: 0}
    cat_type = pd.api.types.CategoricalDtype(categories=[0, 1], ordered=True)

    df = df.replace(dict_repl)

    for col in df.columns:
        if col != 'AGE':
            df[col] = df[col].astype(cat_type)
        
    return df

def rm_outliers(df):
    """remove outliers"""
    df = df[df['AGE'] != 21].reset_index(drop=True)
    
    return df

def split_data(df):
    """split into training and test data"""
    X = df.drop('LUNG_CANCER', axis=1)
    y = df['LUNG_CANCER']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def stdz_features(X_train, X_test):
    """standardize feature variables"""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, scaler
    
def pca_decomp(X_train, X_test):
    """lower dimension"""
    pca = PCA(n_components=2)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    
    return X_train, X_test, pca

def preprocess_data(df):
    """preprocess data for modeling"""
    scaler = None
    pca = None
    
    df = drop_dupl(df)
    df = uni_cols(df)
    df = cat_cols(df)
    df = rm_outliers(df)
    
    X_train, X_test, y_train, y_test = split_data(df)
    X_train, X_test, scaler = stdz_features(X_train, X_test)
#    X_train, X_test, pca = pca_decomp(X_train, X_test)
    
    return X_train, X_test, y_train, y_test, scaler, pca