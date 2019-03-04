# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 16:11:52 2019

@author: Sergei
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


r_data = pd.read_csv('svm-data.csv', index_col=None, header=None)

samples = r_data[r_data.columns[1:3]]
target = r_data[r_data.columns[0]]
#samples[7:8] - certain row from dataframe (in this case - row number 7)

clf = SVC(C=100000, gamma='auto', random_state=241, kernel='linear')
clf.fit(samples, target) 
#print(clf.predict(samples[5:6]))
print(np.sort(clf.support_+1)) # так как нумерация с единицы начинается в задании

sarr = [str(a) for a in np.sort(clf.support_+1)]

f= open("answers_6/1.txt","w+")
#f.write(str(np.sort(clf.support_+1))[1:-1])
f.write(str(" " . join(sarr))) # этот метод не даёт лишних пробелов
f.close()







