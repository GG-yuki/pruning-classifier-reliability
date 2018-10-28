import pandas as pd 

import numpy as np 



data = pd.read_csv('KS_pre_process1.csv') # Reading the dataset from the drive
data.head()
columns_target = ['state'] # target variable - state (Failures, Success)
columns_train = ['category', 'main_category', 'currency', 'goal', 'country', 'duration_days'] # training variables
data = data.dropna() # dropping all null values

X = data[columns_train] 
Y = data[columns_target]

# Counting all null values to confirm
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

# Conversion of string to int

X['category'] = pd.factorize(X.category)[0] + 1 
X.head()
X['main_category'] = pd.factorize(X.main_category)[0] + 1 
X['currency'] = pd.factorize(X.currency)[0] + 1 
X['country'] = pd.factorize(X.country)[0] + 1 
X['main_category'] = pd.factorize(X.main_category)[0] + 1 
X['currency'] = pd.factorize(X.currency)[0] + 1 
X['country'] = pd.factorize(X.country)[0] + 1 

# Performing RandomForest Classifier on the Dataset

from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 42) 

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

RFC = RandomForestClassifier(n_estimators=100)

RFC.fit(X_train, Y_train)

Accuracy_RFC = round(RFC.score(X_test, Y_test) * 100, 2)

print("RandomForest : ", Accuracy_RFC)





