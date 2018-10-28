import pandas as pd 
import numpy as np 
import lightgbm as lgb
from sklearn.metrics import accuracy_score, brier_score_loss, precision_score, balanced_accuracy_score, average_precision_score
from sklearn import neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier

#data = pd.read_csv('KS_original_pre_process.csv')
#data = pd.read_csv('KS_manual_pre_process.csv')
data = pd.read_csv('KS_grubb_pre_processTest1.csv')

ogdf = pd.read_csv('KS_grubb_pre_processTest1.csv') 
print(data.shape)
#print(data[data.goal > 100000])
