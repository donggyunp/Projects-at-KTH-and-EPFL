import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder,MinMaxScaler,normalize,RobustScaler
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
train = pd.read_csv('TrainOnMe.csv')
train = train.drop("Unnamed: 0",axis=1)                                                         
train = train.drop([258,259,260,261,262,263,264,265],axis=0)
train = train.reset_index(drop=True)
train.head()

ev=pd.read_csv('EvaluateOnMe.csv')
ev = ev.drop("Unnamed: 0",axis=1)
ev = ev.reset_index(drop=True)
for i in range(ev['x6'].shape[0]):
    if ev['x6'][i] == 'GMMs and Accordions':
        ev['x6'][i] = 0
    elif ev['x6'][i] == 'Bayesian Inference':
        ev['x6'][i] = 1
for i in range(ev['x12'].shape[0]):
    if ev['x12'][i] == 'False':
        ev['x12'][i] = 0
    elif ev['x12'][i] == 'True':
        ev['x12'][i] = 1





for i in range(train['x6'].shape[0]):
    if not(train['x6'][i] == 'GMMs and Accordions' or train['x6'][i] == 'Bayesian Inference'):
        train=train.drop([i],axis=0)
train = train.reset_index(drop=True)
for i in range(train['x6'].shape[0]):
    if train['x6'][i] == 'GMMs and Accordions':
        train['x6'][i] = 0
    elif train['x6'][i] == 'Bayesian Inference':
        train['x6'][i] = 1

for i in range(train['x12'].shape[0]):
    if not(train['x12'][i] == 'True' or train['x12'][i] == 'False'):
        train=train.drop([i],axis=0)
train = train.reset_index(drop=True)

for i in range(train['x12'].shape[0]):
    if train['x12'][i] == 'False':
        train['x12'][i] = 0
    elif train['x12'][i] == 'True':
        train['x12'][i] = 1
for i in range(train['x1'].shape[0]):
    if np.abs(train['x1'][i])>100:
        train=train.drop([i],axis=0)
train = train.reset_index(drop=True)

'''
train['x1'] = pd.to_numeric(train['x1'], errors='coerce')
train['x2'] = pd.to_numeric(train['x2'], errors='coerce')
train['x3'] = pd.to_numeric(train['x3'], errors='coerce')
train['x4'] = pd.to_numeric(train['x4'], errors='coerce')
train['x5'] = pd.to_numeric(train['x5'], errors='coerce')
train['x6'] = pd.to_numeric(train['x6'], errors='coerce')
train['x7'] = pd.to_numeric(train['x7'], errors='coerce')
train['x8'] = pd.to_numeric(train['x8'], errors='coerce')
train['x9'] = pd.to_numeric(train['x9'], errors='coerce')
train['x10'] = pd.to_numeric(train['x10'], errors='coerce')
train['x11'] = pd.to_numeric(train['x11'], errors='coerce')
train['x12'] = pd.to_numeric(train['x12'], errors='coerce')
'''
train = train.dropna()
train = train.reset_index(drop=True)
#le = LabelEncoder()
y = train.y
#le.fit(y) 
print(y)
'''
for i in range(y.shape[0]):
    if y[i]=='Shoogee':
        y[i]=1
    elif y[i]=='Bob':
        y[i]=2
    elif y[i]=='Atsuto':
        y[i]=3
    elif y[i]=='Jorg':
        y[i]=4 
'''
X = train.drop("y",axis=1)
#onehot=OneHotEncoder()
#onehot.fit(train['x6'],train['x12'])
#onehot.fit(train['x12']) 
print(X)
#sc = StandardScaler()
#X = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)

#sc = RobustScaler()
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#regressor = RandomForestClassifier(n_estimators=15, random_state=0)
#regressor.fit(X_train, y_train)
#y_pred = regressor.predict(X_test)

######
rf_model = RandomForestClassifier(
    n_estimators=1500,
    #criterion='gini',
    criterion='entropy',
    random_state=3000,
    bootstrap=True,
    max_depth=40,
    n_jobs=-1
)
rf_model2 = RandomForestClassifier(
    n_estimators=1500,
    #criterion='gini',
    criterion='entropy',
    random_state=1000,
    bootstrap=True,
    max_depth=30,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
X = sc.fit_transform(X)
ev = sc.transform(ev)
rf_model2.fit(X,y)
rf_predictions = rf_model2.predict(ev)
print(rf_model.score(X_test, y_test))
######
#print(confusion_matrix(y_test,y_pred))
#print(classification_report(y_test,y_pred))
#print(accuracy_score(y_test, y_pred))
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
scores = cross_val_score(rf_model, X, y, scoring='accuracy', cv=cv)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

with open('forest_predictions.txt', 'a') as f:
    labels = rf_predictions
    for label in labels:
        f.write(f'{label}\n')
