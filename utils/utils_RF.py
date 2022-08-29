from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.utils import class_weight


def train_rf_classifier_model(X, y):
    try:
        clf = RandomForestClassifier(max_depth=4, random_state=42, class_weight='balanced').fit(X, y)
        #clf.fit(X, y)
        print('|', end='')
        return clf
    except:
        print('Failed to create the model')
        return None
    
    
    
def apply_rf_classifier_model(clf, X):
    try:
        print('|', end='')
        return clf.predict(X) #[:,0]
       # return clf.predict_proba(X) #[:,0] #if we want the probabilities of beloging to class 0 or 1
    except:
        return None
    
    
def train_rf_regress_model(X, y):
    try:
        rgf =  RandomForestRegressor(max_depth=4, random_state=42).fit(X, y)
        #clf.fit(X, y)
        print('|', end='')
        return rgf
    except:
        print('Failed to create the model')
        return None
    
    
    
def apply_rf_regress_model(rgf, X):
    try:
        print('|', end='')
        return rgf.predict(X) 
    except:
        return None
    
    
# evaluation metrics
def eval_rf_auc(clf, X, y):
    try:
        print('|', end='')
        
        y_pred = clf.predict(X)
        auc = roc_auc_score(y, clf.predict_proba(X)[:,1])    
        return auc
    except:
        return None
    
def eval_rf_precision(clf, X, y):
    try:
        print('|', end='')
        y_pred = clf.predict(X)
        precision=precision_score(y,y_pred) 
        return precision
    except:
        return None
    
def eval_rf_recall(clf, X, y):
    try:
        print('|', end='')
        
        y_pred = clf.predict(X)
        recall=recall_score(y,y_pred)
        return recall
    except:
        return None
    
    
def eval_rf_mse(rfs, X, y):
    try:
        print('|', end='')
        
        y_pred = rfs.predict(X)
        mse=mean_squared_error(y,y_pred)
        return mse
    except:
        return None
    

    
def eval_rf_classifier_model(clf, X, y):
    #not really working 
    try:
        print('|', end='')
        
        y_pred = clf.predict(X)
        auc = roc_auc_score(y, clf.predict_proba(X)[:,1])    
        recall=recall_score(y,y_pred)
        precision=precision_score(y,y_pred)
        results = pd.DataFrame([(auc, recall, precision) ], columns=['auc', 'recall', 'precision'])
       # results = results.to_xarray()
        
        #return np.array([auc,recall,precision],dtype=object)
        return results
    except:
        return None