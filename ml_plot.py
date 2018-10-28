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

columns_target = ['state'] 
columns_train = ['category', 'main_category', 'currency', 'goal', 'country', 'duration_days'] 

data = data.dropna()
#data = data[(data['goal'] <= 100000) & (data['goal'] >= 1000)].copy()

X = data[columns_train] 
Y = data[columns_target]

OX = ogdf[columns_train] 
OY = ogdf[columns_target]

from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42) 
OX_train, OX_test, OY_train, OY_test = train_test_split(OX,OY, test_size = 0.10, random_state = 42) 


a = list()
for i in range(10):
    linear_svc = LinearSVC()
    linear_svc.fit(X_train, np.ravel(Y_train)) 
    print("Score ", linear_svc.score(OX_test, OY_test))
    a.append(linear_svc.score(OX_test, OY_test))

    mlp = MLPClassifier(solver='adam', 
                        alpha=1e-5, 
                        hidden_layer_sizes=(21, 2), 
                        random_state=1)
    mlp.fit(X_train, np.ravel(Y_train)) 
    print("Score ", mlp.score(OX_test, OY_test))
    a.append(mlp.score(OX_test, OY_test))



    clf = LogisticRegression()
    clf.fit(X_train, np.ravel(Y_train)) 
    print("Score ", clf.score(OX_test, OY_test))
    a.append(clf.score(OX_test, OY_test))



    #print("Precision Score",precision_score(OY_test, y_pred, average='macro'))
    #print("Balanced Accuracy Score", balanced_accuracy_score(OY_test, y_pred))

    knn=neighbors.KNeighborsClassifier()
    knn.fit(X_train, np.ravel(Y_train)) 
    print("Score ", knn.score(OX_test, OY_test))
    a.append(knn.score(OX_test, OY_test))


    #print("Precision Score",precision_score(OY_test, y_pred, average='macro'))
    #print("Balanced Accuracy Score", balanced_accuracy_score(OY_test, y_pred))


    rfc = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    rfc.fit(X_train, np.ravel(Y_train)) 
    print("Score ", rfc.score(OX_test, OY_test))
    a.append(rfc.score(OX_test, OY_test))


    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, np.ravel(Y_train)) 
    print("Score ", decision_tree.score(OX_test, OY_test))
    a.append(decision_tree.score(OX_test, OY_test))



    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                             algorithm="SAMME",
                             n_estimators=200)


    bdt.fit(X_train, np.ravel(Y_train)) 
    print("Score ", bdt.score(OX_test, OY_test))
    a.append(bdt.score(OX_test, OY_test))




    bagging = BaggingClassifier(
        neighbors.KNeighborsClassifier(
            n_neighbors=8,
            weights='distance'
            ),
        oob_score=True,
        max_samples=0.5,
        max_features=1.0
        )
    bagging.fit(X_train, np.ravel(Y_train)) 
    print("Score ", bagging.score(OX_test, OY_test))
    a.append(bagging.score(OX_test, OY_test))
    
    
print 'original array:'
print a
aa = np.array(a).reshape(10,8)

print(aa.mean(axis=0))
