# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 17:26:37 2019

@author: Nazarov
"""

import pandas as pd
import numpy as np
from collections import Counter
import regex as re
import matplotlib.pyplot as plt

data = pd.read_csv('titanic.csv', index_col='PassengerId')

#print(data['Sex'])
#fem_all = data[data.Sex=='female']
#data.Name.str.contains('Master', regex=False)
#print(fem_all['Name'])
#len(fem_all.index) # length array
#data.Survived.shape[0] # length array Survived
#data.Survived.count() # length array Survived
#fem_count = fem_all.shape[0] # length dataset
#male_all = data[data.Sex=='male']
#male_count = male_all.shape[0] # length dataset
#data[(data.Survived==1) & (data.Pclass==1)] # select for two conditions
#data.Age.dropna() # non missing value
#data.Age[data.Age.notnull()]  # non missing value too
#data1=data[['SibSp','Parch']] # selecting certain column from array 
#data[['SibSp','Parch']].corr().iat[0,1] # select certain value from array
#fe1 = fem.Name.str.replace(r'^\w+.\s+\w+.\s', '')
#rr=data[data.Name.str.contains('Miss', regex=False) & data.Sex.str.contains('female', regex=False)]

# !! https://pandas.pydata.org/pandas-docs/stable/text.html

#1
f= open("answers_1/1.txt","w+")
f.write(str(data[data.Sex=='male'].shape[0])+" "+str(data[data.Sex=='female'].shape[0]))
f.close()

#2
x1 = data[data.Survived==1].shape[0] # survived passangers
x2 = data.Survived.count() # all passangers
x3 = 100*x1/x2
#print(int(x3.round(0)))
f= open("answers_1/2.txt","w+")
f.write((str(x3.round(2))))
f.close()

#21 - for experience
a11 = data.Survived.count() # all passangers
print("all passangers: "+str(a11))
s11 = data[data.Survived==1].shape[0] # all survived passangers
print("all survived passangers: "+str(s11))
c11 = data[(data.Survived==1) & (data.Pclass==1)].shape[0] # survived passangers from 1 class
nc11 = data[(data.Pclass==1)].shape[0] # passangers from 1 class
pc11 = int(100*c11/nc11)
print("survived passangers 1 class: "+str(pc11)+" %")

c22 = data[(data.Survived==1) & (data.Pclass==2)].shape[0] # survived passangers from 1 class
nc22 = data[(data.Pclass==2)].shape[0] # passangers from 2 class
pc22 = int(100*c22/nc22)
print("survived passangers 2 class: "+str(pc22)+" %")

c33 = data[(data.Survived==1) & (data.Pclass==3)].shape[0] # survived passangers from 1 class
nc33 = data[(data.Pclass==3)].shape[0] # passangers from 3 class
pc33 = int(100*c33/nc33)
print("survived passangers 3 class: "+str(pc33)+" %")

#3
xx3 = 100*nc11/a11
f= open("answers_1/3.txt","w+")
f.write((str(xx3.round(2))))
f.close()

#4
mn = data.Age.dropna().mean() # mean age
md = data.Age.dropna().median() # mediana age
f= open("answers_1/4.txt","w+")
f.write(str(round(mn,2))+" "+str(round(md,2)))
f.close()

#5
f= open("answers_1/5.txt","w+")
f.write(str(data[['SibSp','Parch']].corr().iat[0,1].round(2)))
f.close()

#6
#apply(lambda
#np.where(data.Name.str.contains('Master'))
#miss = data[data.Name.str.contains('Miss', regex=False) & data.Sex.str.contains('female', regex=False)].Name.str.replace(r'^\w+.\s+\w+.\s', '') # remove not use words
#mrs = data[data.Name.str.contains('Mrs', regex=False) & data.Sex.str.contains('female', regex=False)]
#fem = data[data.Sex.str.contains('female', regex=False)].Name.str.replace(r'^\w+.\s+\w+.\s', '') # remove not use words
fem = data[data.Sex.str.contains('female', regex=False)].Name.str.replace(r'(.*)[(]', '') # remove all to brackets
fem = fem.str.replace(r'(.*)(Mrs.)\s', '') # remove all disturbing characters
fem = fem.str.replace(r'(.*)(Miss.)\s', '') # remove all disturbing characters
fem = fem.str.replace(r'[")]', '') # remove all disturbing characters
#fem = fem.str.replace(r'\s(.*)', '') # select first word

# count frequence v1:
A = np.array(fem)
B = []
for i in range(len(fem)):
    X = A[i].split()
    B.extend(X)

unique,pos = np.unique(B,return_inverse=True)
counts = np.bincount(pos)
maxpos = counts.argmax()
fr_name_1 = (unique[maxpos],counts[maxpos])[0]
f= open("answers_1/6.txt","w+")
f.write(fr_name_1)
f.close()



#print(data)