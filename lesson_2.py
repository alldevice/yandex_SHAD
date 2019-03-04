# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 14:30:38 2019

@author: Sergei
"""

import pandas as pd
import numpy as np
#from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import graphviz

data = pd.read_csv('titanic.csv', index_col='PassengerId')
data0 = data[['Pclass','Fare','Age','Survived']] # besides Sex
le = LabelEncoder() # for transform str to int (column 'Sex')
#data0.insert(0,'Sex',le.fit_transform(data.Sex)) # insert transform column 'Sex' v1
data0.insert(0,'Sex',label_binarize(data.Sex, classes=['female', 'male'])) # insert transform column 'Sex' v2
data1 = data0.dropna(subset=['Pclass','Fare','Age','Sex','Survived'],axis='rows')
samples = data1.drop(['Survived'], axis=1)
target = data1[['Survived']]
#le.inverse_transform(data1.Sex) # for back


clf = tree.DecisionTreeClassifier(random_state=241)
clf = clf.fit(samples, target)


feature_names = list(samples.columns.values) # get column names
target_names = np.array(['not_surv','surv'])

#dot_data = tree.export_graphviz(clf, out_file=None)
dot_data = tree.export_graphviz(clf, out_file=None,feature_names=feature_names,class_names=target_names,filled=True, rounded=True,special_characters=None)
graph = graphviz.Source(dot_data)
graph.render("titanic")
#graph 

# Exercise
importances = clf.feature_importances_ 
first_im = np.argsort(importances)[-1] # last index (maximum)
second_im = np.argsort(importances)[-2] # penultimate index
f= open("answers_2/1.txt","w+")
f.write(feature_names[first_im]+' '+feature_names[second_im])
f.close()
