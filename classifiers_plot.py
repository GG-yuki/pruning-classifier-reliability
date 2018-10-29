import pandas as pd 
import numpy as np 
from sklearn.metrics import accuracy_score
from sklearn import neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

#data = pd.read_csv('KS_original_pre_process.csv')
#data = pd.read_csv('KS_manual_pre_process.csv')
data = pd.read_csv('KS_grubb_pre_process.csv')
ogdf = pd.read_csv('KS_original_pre_process.csv')
print(data.shape)

columns_target = ['state'] 
columns_train = ['category', 'main_category', 'currency', 'goal', 'country', 'duration_days']
data = data.dropna()

X = data[columns_train] 
Y = data[columns_target]
OX = ogdf[columns_train] 
OY = ogdf[columns_target]
from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42) 
OX_train, OX_test, OY_train, OY_test = train_test_split(OX,OY, test_size = 0.10, random_state = 42)
itr = 1

a = list()
for i in range(itr):
    linear_svc = LinearSVC()
    linear_svc.fit(X_train, np.ravel(Y_train)) 
    a.append(linear_svc.score(OX_test, OY_test))
    mlp = MLPClassifier(solver='adam', 
                        alpha=1e-5, 
                        hidden_layer_sizes=(21, 2), 
                        random_state=1)
    mlp.fit(X_train, np.ravel(Y_train)) 
    a.append(mlp.score(OX_test, OY_test))

    clf = LogisticRegression()
    clf.fit(X_train, np.ravel(Y_train)) 
    a.append(clf.score(OX_test, OY_test))

    knn=neighbors.KNeighborsClassifier()
    knn.fit(X_train, np.ravel(Y_train)) 
    a.append(knn.score(OX_test, OY_test))

    rfc = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    rfc.fit(X_train, np.ravel(Y_train)) 
    a.append(rfc.score(OX_test, OY_test))

    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, np.ravel(Y_train)) 
    a.append(decision_tree.score(OX_test, OY_test))

    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                             algorithm="SAMME",
                             n_estimators=200)
    bdt.fit(X_train, np.ravel(Y_train)) 
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
    a.append(bagging.score(OX_test, OY_test))

aa = np.array(a).reshape(itr,8)
bb = aa.mean(axis=0)
print(aa.mean(axis=0))
fig, ax = plt.subplots()
objects = ('SVC', 'MLP', 'LGR', 'KNN', 'RF', 'DT', 'AB','BC')
y_pos = np.arange(len(objects))
performance = [bb[0],bb[1],bb[2],bb[3],bb[4],bb[5],bb[6],bb[7]]
bar = plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy')
plt.title('Machine Learning Algorithms (Grubbs Test Pruning)')
cnt = 0
for rect in bar:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., 1.02*height,
            '%s' % str("{0:.2f}".format(100*performance[cnt])) + "%", ha='center', va='bottom')
    cnt += 1
plt.ylim([0, 1])
plt.savefig('graphGrubbs.png')  
plt.show()