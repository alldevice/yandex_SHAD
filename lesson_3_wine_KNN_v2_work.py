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

data = pd.read_csv('wine.data', index_col=None, header=None)

target_class_names = np.array(['one','two','three'])
target_class0 = data.loc[:,0] # get first column with class features
target_class = target_class0.values.tolist() # output list - not dataframe
features0 = data.loc[:,1:] # get all features
features = features0.values.tolist() # output list - not dataframe
#features.iloc[1] # get one row

k = 1

# v1 - simple split:
X_train, X_test, y_train, y_test = train_test_split(features, target_class, test_size=0.2, random_state=42, shuffle=True)
#then fit:
neigh_1 = KNeighborsClassifier(n_neighbors=k) # it's model KNN
#neigh = NearestNeighbors(n_neighbors=5)
neigh_1.fit(X_train, y_train) #
#predictions_1 = neigh_1.predict(X_test)
#plt.scatter(y_test, predictions_1)
#plt.xlabel('True Values')
#plt.ylabel('Predictions')
mean_score_1 = neigh_1.score(X_test, y_test).mean()
print ('Score without CV:', mean_score_1)

# v2 - split for cross-validation Kfold:
neigh_2 = KNeighborsClassifier(n_neighbors=k) # it's model KNN
n_folds = 5
kf = KFold(n_splits=n_folds, random_state=42, shuffle=True)
mean_score_2 = cross_val_score(neigh_2, features, target_class, scoring='accuracy', cv=kf).mean()
print ('Score with CV:', mean_score_2)

#rr = np.reshape(features[60], (-1, 1)).T.tolist() # reshape - transponding
#print(neigh.predict(rr))
#print(neigh.predict_proba(rr))




  
