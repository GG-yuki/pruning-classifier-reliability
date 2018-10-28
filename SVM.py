#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 


# In[2]:


data = pd.read_csv('KS_pre_process1.csv') 


# In[3]:


columns_target = ['state'] #setting a target column
columns_train = ['category', 'main_category', 'currency', 'goal', 'country', 'duration_days'] 


# In[4]:


data = data.dropna()
data = data[(data['goal'] <= 100000) & (data['goal'] >= 1000)].copy()


# In[5]:


X = data[columns_train] 
Y = data[columns_target]


# In[6]:


X['category'].isnull().sum() 


# In[7]:


X['main_category'].isnull().sum()


# In[8]:


X['currency'].isnull().sum() 


# In[9]:


X['goal'].isnull().sum() 


# In[10]:


X['country'].isnull().sum() 


# In[11]:


X['duration_days'].isnull().sum() 


# In[12]:


X['category'].unique() 


# In[13]:


X.head()


# In[14]:


X['category'].nunique() 


# In[15]:


X['main_category'].nunique() 


# In[16]:


X['currency'].nunique() 


# In[17]:


X['country'].nunique() 


# In[18]:


X.head() 


# In[ ]:


X['category'] = pd.factorize(X.category)[0] + 1 


# In[ ]:


X.head()


# In[ ]:


X['main_category'] = pd.factorize(X.main_category)[0] + 1 


# In[ ]:


X['currency'] = pd.factorize(X.currency)[0] + 1 


# In[ ]:


X['country'] = pd.factorize(X.country)[0] + 1 


# In[ ]:


X['main_category'] = pd.factorize(X.main_category)[0] + 1 


# In[ ]:


X['currency'] = pd.factorize(X.currency)[0] + 1 


# In[ ]:


X['country'] = pd.factorize(X.country)[0] + 1 


# In[ ]:


from sklearn.model_selection import train_test_split 


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 42) 


# In[ ]:


from sklearn import svm 


# In[ ]:


clf = svm.LinearSVC(max_iter=1000)


# In[ ]:


print(clf)


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


fit = clf.fit(X_train, np.ravel(Y_train)) 


# In[ ]:


y_pred = clf.predict(X_test) 


# In[ ]:


print("Linear SVM: ", accuracy_score(Y_test, y_pred))


# In[ ]:


clf = svm.SVC(gamma=1)


# In[ ]:


fit = clf.fit(X_train, np.ravel(Y_train)) 


# In[ ]:


y_pred = clf.predict(X_test) 


# In[ ]:


print("RBF SVM: ", accuracy_score(Y_test, y_pred))

