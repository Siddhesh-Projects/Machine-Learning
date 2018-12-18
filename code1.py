# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 14:29:48 2018

@author: lenovo pc
"""

import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

datafile1 = 'titanic.csv'
datafile2 = 'Wine.csv'
datafile3 = 'Summary of Weather.csv'
#datafile4 = 'Online Retail.csv'

tit_data = pd.read_csv(datafile1)
win_data = pd.read_csv(datafile2)
wea_data = pd.read_csv(datafile3)
#onl_data = pd.read_csv(datafile4)

#print(tit_data.head())
scale(win_data)

m = {'male' : 1, 'female' : 0}
tit_data["Sex"] = tit_data.Sex.map(m)

row1,col1 = tit_data.shape
row2,col2 = win_data.shape
#row4,col4 = onl_data.shape

mark1 = (2*row1)/3
mark2 = (2*row2)/3

tit_train = tit_data[1:int(mark1)][0:row1]
tit_test = tit_data[int(mark1):][0:row1]

#print(tit_data)
#print(tit_train)

win_train = win_data[0:int(mark2)][0:row2]
win_test = win_data[int(mark2):][0:row2]

#print(win_train)

attributes = ["Sex","Age","Siblings/Spouses Aboard"]
#print(attributes["Sex"])

#print(mark1)

#print(tit_data[int(mark1):][0:row1])

#Classification using Random Forest
def RandomForest():
    x,y = tit_train[attributes],tit_train.Survived
    
    #x["Sex"] = x.Sex.map(m)
    
    #print(x)
    #print(y)
    
    random_model = RandomForestClassifier(max_depth=4)
    validate = cross_validation.cross_val_score(random_model,x,y,cv=5)
    
    print(validate.mean())
    
    random_model.fit(x,y)
    
    prediction = random_model.predict(tit_test[attributes])
    
    print(prediction)
    
#RandomForest()

#Clustering using K-Means    
def K_Means():
    cluster = KMeans(n_clusters=3)
    cluster.fit(win_data)
    
    color = np.array(['green','red','blue','yellow'])
    
    plt.scatter(x=win_data.Alcohol,y=win_data.Color_Intensity,c=color[win_data.Customer_Segment])
    
    plt.title('K-Means Classificatipn')
    plt.ylabel('Color Intensity')
    plt.xlabel('Alcohol')

#K_Means()


print(type(wea_data))
wea_data = wea_data.replace('T',0)
wea_data.fillna(0,inplace=True)

print(wea_data)

#transactions = []
#print(row4)
#for i in range(0,row4):






























