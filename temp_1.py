# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 21:14:16 2019

@author: Nazarov
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler
data = [[0, 0], [0, 0], [1, 1], [1, 1]]
scaler = StandardScaler()
#print(scaler.fit(data))

#print(scaler.mean_)

print(scaler.fit_transform(data))




#print(scaler.transform([[2, 2]]))




