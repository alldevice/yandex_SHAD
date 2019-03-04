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

#clf.fit(X, y)
#redictions = clf.predict(X)

arr_test = pd.read_csv('perceptron-test.csv', index_col=None, header=None)
arr_train = pd.read_csv('perceptron-train.csv', index_col=None, header=None)

samples_train = arr_train[arr_train.columns[1:3]]
target_train = arr_train[arr_train.columns[0]]
samples_test = arr_test[arr_test.columns[1:3]]
target_test = arr_test[arr_test.columns[0]]

clf = Perceptron(random_state=241)

def func_perc(X_train, y_train,X_test, y_test):
  clf.fit(X_train, y_train)
  target_test_pred = clf.predict(X_test)
  accur_score = accuracy_score(y_test, target_test_pred)
  return accur_score

# tran without scale
X_train = samples_train
y_train = target_train
X_test = samples_test
y_test = target_test
accur_score_1 = func_perc(X_train, y_train,X_test, y_test)
print('accuracy_score_1: ', accur_score_1)

# tran with scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(samples_train, target_train)
X_test_scaled = scaler.transform(samples_test) # for test data - ONLY just transform
accur_score_2 = func_perc(X_train_scaled, y_train,X_test_scaled, y_test)
print('accuracy_score_2: ', accur_score_2)

f= open("answers_5/1.txt","w+")
f.write(str(np.round((accur_score_2 - accur_score_1), decimals=3)))
f.close()







