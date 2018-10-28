import pandas as pd 
import numpy as np 
data = pd.read_csv('KS_pre_process1.csv') 
data.head()
columns_target = ['state'] 
columns_train = ['category', 'main_category', 'currency', 'goal', 'country', 'duration_days'] 

data = data.dropna()
data = data[(data['goal'] <= 100000) & (data['goal'] >= 1000)].copy()

X = data[columns_train] 
Y = data[columns_target]

X['category'].isnull().sum() 
X['main_category'].isnull().sum()
X['currency'].isnull().sum() 
X['goal'].isnull().sum() 
X['country'].isnull().sum() 
X['duration_days'].isnull().sum() 
X['category'].unique() 
X['category'].nunique() 
X['main_category'].nunique() 
X['currency'].nunique() 
X['country'].nunique() 
X.head() 
X['category'] = pd.factorize(X.category)[0] + 1 
X.head()
X['main_category'] = pd.factorize(X.main_category)[0] + 1 
X['currency'] = pd.factorize(X.currency)[0] + 1 
X['country'] = pd.factorize(X.country)[0] + 1 
X['main_category'] = pd.factorize(X.main_category)[0] + 1 
X['currency'] = pd.factorize(X.currency)[0] + 1 
X['country'] = pd.factorize(X.country)[0] + 1 
from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 42) 
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
print(clf)
from sklearn.metrics import accuracy_score
fit = clf.fit(X_train, np.ravel(Y_train)) 
y_pred = clf.predict(X_test) 
print("LogisticRegression: ", accuracy_score(Y_test, y_pred))

