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
import matplotlib.pyplot as plt


#data = pd.read_csv('KS_original_pre_process.csv')

#data = pd.read_csv('KS_manual_pre_process.csv')
data = pd.read_csv('KS_grubb_pre_process.csv')

ogdf = pd.read_csv('KS_original_pre_process.csv') 

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
    #print("Score ", linear_svc.score(OX_test, OY_test))
    a.append(linear_svc.score(OX_test, OY_test))

    mlp = MLPClassifier(solver='adam', 
                        alpha=1e-5, 
                        hidden_layer_sizes=(21, 2), 
                        random_state=1)
    mlp.fit(X_train, np.ravel(Y_train)) 
    #print("Score ", mlp.score(OX_test, OY_test))
    a.append(mlp.score(OX_test, OY_test))



    clf = LogisticRegression()
    clf.fit(X_train, np.ravel(Y_train)) 
    #print("Score ", clf.score(OX_test, OY_test))
    a.append(clf.score(OX_test, OY_test))



    #print("Precision Score",precision_score(OY_test, y_pred, average='macro'))
    #print("Balanced Accuracy Score", balanced_accuracy_score(OY_test, y_pred))

    knn=neighbors.KNeighborsClassifier()
    knn.fit(X_train, np.ravel(Y_train)) 
    #print("Score ", knn.score(OX_test, OY_test))
    a.append(knn.score(OX_test, OY_test))


    #print("Precision Score",precision_score(OY_test, y_pred, average='macro'))
    #print("Balanced Accuracy Score", balanced_accuracy_score(OY_test, y_pred))


    rfc = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    rfc.fit(X_train, np.ravel(Y_train)) 
    #print("Score ", rfc.score(OX_test, OY_test))
    a.append(rfc.score(OX_test, OY_test))


    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, np.ravel(Y_train)) 
    #print("Score ", decision_tree.score(OX_test, OY_test))
    a.append(decision_tree.score(OX_test, OY_test))



    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                             algorithm="SAMME",
                             n_estimators=200)


    bdt.fit(X_train, np.ravel(Y_train)) 
    #print("Score ", bdt.score(OX_test, OY_test))
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
    #print("Score ", bagging.score(OX_test, OY_test))
    a.append(bagging.score(OX_test, OY_test))

print 'original array:'
print a
aa = np.array(a).reshape(10,8)
bb=aa.mean(axis=0)
print(bb.shape)

print(aa.mean(axis=0))

#frequencies = [bb[0],bb[1],bb[2],bb[3],bb[4],bb[5],bb[6],bb[7]]
# In my original code I create a series and run on that, 
# so for consistency I create a series from the list.
#freq_series = pd.Series.from_array(frequencies)

#x_labels = ['SVC', 'MLP', 'LGR', 'KNN', 'RF', 'DT', 'AB','BC']

#plt.figure(figsize=(12, 8))
#ax = freq_series.plot(kind='bar')
#ax.set_title('Machine Learning Algorithms')
#ax.set_xlabel('Amount ($)')
#ax.set_ylabel('Accuracy')
#ax.set_xticklabels(x_labels)

#rects = ax.patches

# Make some labels.
#labels = ["label%d" % i for i in xrange(len(rects))]

#for rect, label in zip(rects, labels):
 #   height = rect.get_height()
  #  ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label,
   #         ha='center', va='bottom')

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

        
#print 'modified array:'
#print newlist
#avg = list()
#for i in range(8):
#    print('avg ',newlist.sum(axis=1)/10.0)
    #avg[i] = newlist.sum(axis=i)/10
#print avg

#averages = list()
#for i in range(5):
 #   for j in range(i+59):    
  #      sum = sum + a[j]
   #     j+6
    #averages[i] = sum


