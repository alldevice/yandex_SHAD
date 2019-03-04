# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 16:11:52 2019

@author: Sergei
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

boston = load_boston()

X_data = preprocessing.scale(boston.data)
target_class = boston.target


#x = np.linspace(1,10,num=200)
n_folds = 5
iter_x = np.linspace(1,10,num=200)
mean_score = []
for p in iter_x:
  neigh = KNeighborsRegressor(n_neighbors=5, weights='distance', algorithm='auto', leaf_size=30, p=p, metric='minkowski', metric_params=None, n_jobs=None)
  kf = KFold(n_splits=n_folds, random_state=42, shuffle=True)
  mean_score.append(cross_val_score(neigh, X_data, target_class, scoring='neg_mean_squared_error', cv=kf))
 
#print(mean_score[0])
nmse = np.mean(mean_score[::],axis=1)
ans = np.argmax(nmse)+1
#print(np.mean(mean_score[::],axis=1))

# Draw plot
fig = plt.figure()
plt.grid(True)
plt.plot(iter_x, nmse)
plt.show()

f= open("answers_4/1.txt","w+")
f.write(str(ans))
f.close()


