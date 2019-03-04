# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 16:11:52 2019

@author: Sergei
"""

import pandas as pd
import numpy as np
#from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from sklearn import preprocessing

data = pd.read_csv('wine.data', index_col=None, header=None)

target_class_names = np.array(['one','two','three'])
target_class0 = data.loc[:,0] # get first column with class features
target_class = target_class0.values.tolist() # output list - not dataframe
features0 = data.loc[:,1:] # get all features
features = features0.values.tolist() # output list - not dataframe


def func_exercise(X_data):
  n_folds = 5
  iterat = 50
  mean_score = []
  for k in np.arange(0,iterat):
    neigh_2 = KNeighborsClassifier(n_neighbors=k+1) # it's model KNN
    kf = KFold(n_splits=n_folds, random_state=42, shuffle=True)
    mean_score.append(cross_val_score(neigh_2, X_data, target_class, scoring='accuracy', cv=kf))
  return mean_score

dd0 = func_exercise(features)
mean_score_2 = np.mean(func_exercise(features),axis=1)
a1 = np.argmax(mean_score_2)+1
a2 = np.max(mean_score_2).round(2)
print ('value k for max score: ', a1,'; max score: ', a2)
f= open("answers_3/1.txt","w+")
f.write(str(a1))
f.close()

f= open("answers_3/2.txt","w+")
f.write(str(a2))
f.close()

features_3 = preprocessing.scale(features)
dd = func_exercise(features_3)
mean_score_4 = np.mean(func_exercise(features_3),axis=1)
a3 = np.argmax(mean_score_4)+1
a4 = np.max(mean_score_4).round(2)
print ('value k for max score: ', a3,'; max score: ', a4)
f= open("answers_3/3.txt","w+")
f.write(str(a3))
f.close()

f= open("answers_3/4.txt","w+")
f.write(str(a4))
f.close()




#m = max(kMeans)
#indices = [i for i, j in enumerate(kMeans) if j == m]

#rr = np.reshape(features[60], (-1, 1)).T.tolist() # reshape - transponding
#print(neigh.predict(rr))
#print(neigh.predict_proba(rr))




  
