# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 16:11:52 2019

@author: Sergei
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier


data = pd.read_csv('wine.data', index_col=None, header=None)

target_class_names = np.array(['one','two','three'])
target_class0 = data.loc[:,0] # get first column with class features
target_class = target_class0.values.tolist() # output list - not dataframe
features0 = data.loc[:,1:] # get all features
features = features0.values.tolist() # output list - not dataframe
#features.iloc[1] # get one row

#neigh = NearestNeighbors(n_neighbors=5)
neigh = KNeighborsClassifier(n_neighbors=7)
neigh.fit(features, target_class)

#features = df[3]
rr = np.reshape(features[170], (-1, 1)).T.tolist() # reshape - transponding
print(neigh.predict(rr))
print(neigh.predict_proba(rr))



  
