import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from xgboost import XGBClassifier
import os 
from time import sleep
if 'resultfile.csv' in os.listdir():
    os.remove("resultfile.csv")
a=pd.read_csv('train.csv')
b=pd.read_csv('test.csv')
a = a.drop(['PassengerId','Name','Ticket','Cabin'], axis = 1)
b = b.drop(['Name','Ticket','Cabin'], axis = 1)
a = a.fillna({"Embarked": "S"})
X_train = a.iloc[:, 1:].values
y_train = a.iloc[:, 0].values
X_test = b.iloc[:, 1:].values

"""sleep(2)"""

from sklearn.impute import SimpleImputer
i = SimpleImputer(missing_values=np.nan, strategy='mean')
X_train[:,2:6]=i.fit_transform(X_train[:,2:6])
X_test[:,2:6]=i.fit_transform(X_test[:,2:6])

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
X_train[:, 1] = le.fit_transform(X_train[:, 1])
X_train[:, 6] = le.fit_transform(X_train[:, 6])
X_test[:, 1] = le.fit_transform(X_test[:, 1])
X_test[:, 6] = le.fit_transform(X_test[:, 6])
print(X_train[:,6])
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
randomforest = RandomForestClassifier(n_estimators=200)
randomforest.fit(X_train, y_train)

# xgb = XGBClassifier()
# xgb.fit(X_train, y_train)

ids = b['PassengerId']
predictions = randomforest.predict(X_test)
output = pd.DataFrame({'PassengerId': ids, 'Survived': predictions})
output.to_csv('resultfile.csv', index=False)

# a=a.drop(['Name','PassengerId','Fare'],axis=1)
#print(tabulate(X_test))
#print(a['Survived'].tolist())
#print(a.info())

