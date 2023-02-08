from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def lr_model(X_train, y_train):
    """Linear Regression model"""
    lr = LogisticRegression()
    param_grid = {
        'C': [0.1, 1],
        'penalty': ['l1', 'l2'],
        'solver': ['sag', 'saga']
    }
    
    lr = best_params(lr, param_grid, X_train, y_train)
    
    return lr

def svm_model(X_train, y_train):
    """Support Vector Machine model"""
    svm = SVC(probability=True)
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': [0.001, 0.01, 0.1, 'scale']
    }
    
    svm = best_params(svm, param_grid, X_train, y_train)
    
    return svm

def rf_model(X_train, y_train):
    """Random Forest model"""
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt']
    }
    
    rf = best_params(rf, param_grid, X_train, y_train)
    
    return rf

def knn_model(X_train, y_train):
    """K-Nearest Neighbors model"""
    knn = KNeighborsClassifier()
    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }
    
    knn = best_params(knn, param_grid, X_train, y_train)
    
    return knn
    
def best_params(model, params, X_train, y_train):
    """find parameters with best performance"""
    grid_search = GridSearchCV(estimator=model, param_grid=params, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    return best_model

def train_model(X_train, y_train):
    """train models"""
    lr = None
    svm = None
    rf = None
    knn = None
    
    lr = lr_model(X_train, y_train)
#    svm = svm_model(X_train, y_train)
#    rf = rf_model(X_train, y_train)
#    knn = knn_model(X_train, y_train)
    
    return lr, svm, rf, knn

def eval_model(model_list, X_test, y_test):
    """evaluate models and select model with highest accuracy"""
    best_model = None
    best_acc = 0
    
    for mdl in model_list:
        if mdl != None:
            y_pred = mdl.predict(X_test)
            accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
#            print('accuracy:', accuracy, '%')
            
            if accuracy > best_acc:
                best_acc = accuracy
                best_model = mdl
    
    return best_model
    
def data_modeling(X_train, X_test, y_train, y_test):
    """integrate train_model and eval_model functions"""
    model_list = train_model(X_train, y_train)
    best_model = eval_model(model_list, X_test, y_test)
    
    return best_model