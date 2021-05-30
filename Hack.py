import pandas as pd
import numpy as np
import io
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import KFold , StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
warnings.filterwarnings("ignore")
# Import Data

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


train['source'] = 'train'
test['source'] = 'test'

df = pd.concat([train, test])

# Preprocessing  

df['Stay'] = df['Stay'].replace({'0-10' : 0,
                                 '11-20' : 1,
                                 '21-30': 2,
                                 '31-40': 3,
                                 '41-50': 4,
                                 '51-60': 5,
                                 '61-70': 6, 
                                 '71-80': 7,
                                 '81-90': 8,
                                 '91-100': 9,
                                 'More than 100 Days' : 10
                                     })

df['Hospital_type_code'] = df['Hospital_type_code'].replace({'a' : 0, 'b' : 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6 })

df['Hospital_region_code'] = df['Hospital_region_code'].replace({'X' : 0, 'Y' : 1, 'Z': 2})

dept = pd.get_dummies(df['Department'])
df = pd.concat([df, dept], axis = 1)

df['Ward_Type'] = df['Ward_Type'].replace({'R' : 0, 'Q' : 1, 'S' : 2, 'P' : 3, 'T' : 4, 'U': 5})

df['Ward_Facility_Code'] = df['Ward_Facility_Code'].replace({'A' : 0, 'B' : 1, 'C' : 2, 'D' : 3, 'E' : 4, 'F': 5})

df['Bed Grade'] = df['Bed Grade'].transform(lambda x: x.fillna('2.0'))

adtype = pd.get_dummies(df['Type of Admission'])
df = pd.concat([df, adtype], axis = 1)

sev = pd.get_dummies(df['Severity of Illness'])
df = pd.concat([df, sev], axis = 1)

df['Age'] = df['Age'].replace({'0-10' : 0, 
                               '11-20' : 1,
                               '21-30': 2,
                               '31-40': 3,
                               '41-50': 4,
                               '51-60': 5,
                               '61-70': 6, 
                               '71-80': 7,
                               '81-90': 8,
                               '91-100': 9,
                                     })

df['City_Code_Patient'] = df.groupby(['Hospital_type_code', 'City_Code_Hospital', 'Hospital_region_code', 'Hospital_code']).City_Code_Patient.transform(lambda x: x.fillna(x.median()))

# Delete Unwanted Columns

del df['Department']
del df['Type of Admission'] 
del df['Severity of Illness'] 

Train = df[df['source'] == 'train']
Test = df[df['source'] == 'test']

del Train['source']
del Test['source']
del Test['Stay']
del Train['case_id']
del Test['case_id']

yTrain = Train.pop('Stay')

sc = StandardScaler()
sTrain = sc.fit_transform(Train)
sTest = sc.fit_transform(Test)

model_params = {
    'LGBM': {
        'model': LGBMClassifier(),
        'params' : {
            'n_estimators': [10, 100, 200, 500],
            'num_leaves': [10, 50, 100, 200, 500],
            'max_depth': [5, 10, 15, 20, 25]
            
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [50, 100, 150, 500],
            'min_samples_leaf': [1, 2, 3, 4],
            'max_depth': [5, 10, 15, 20, 25]
        }
    },
     'cat': {
        'model': CatBoostClassifier(eval_metric='Accuracy'),
        'params' : {
            'n_estimators': [10, 100, 200, 500],
            'max_depth': [5, 10, 15, 20, 25] 
        }
    },
       'xgb': {
        'model': XGBClassifier(),
        'params' : {
            'n_estimators': [10, 100, 200, 500],
            'max_depth': [5, 10, 15, 20, 25],
            'min_child_weight' : [ 1, 3, 5, 7 ],
            'gamma'            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
            'colsample_bytree' : [ 0.3, 0.4, 0.5 , 0.7 ]
        }
    },
}

scores = []

for model_name, mp in model_params.items():
    clf = RandomizedSearchCV(mp['model'], mp['params'], cv=3, return_train_score=False, verbose = 1 )
    clf.fit(sTrain, yTrain)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })


























