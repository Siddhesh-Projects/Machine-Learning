# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 14:29:48 2018

@author: lenovo pc
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn import linear_model
from apyori import apriori

datafile1 = 'titanic.csv'
datafile2 = 'Wine.csv'
datafile3 = 'Summary of Weather.csv'
datafile4 = 'marketbasket.csv'

tit_data = pd.read_csv(datafile1)
win_data = pd.read_csv(datafile2)
wea_data = pd.read_csv(datafile3)
mar_data = pd.read_csv(datafile4)

#print(tit_data.head())
scale(win_data)

#print(type(wea_data))
wea_data = wea_data.replace('T',0)
wea_data = wea_data.replace('#VALUE!',0)
wea_data.fillna(0,inplace=True)
#print(np.sum(np.invert(np.isreal(wea_data[attributes1]))))

m = {'male' : 1, 'female' : 0}
tit_data["Sex"] = tit_data.Sex.map(m)

row1,col1 = tit_data.shape
row2,col2 = win_data.shape
row3,col3 = wea_data.shape
row4,col4 = mar_data.shape

mark1 = (2*row1)/3
mark2 = (2*row2)/3
mark3 = (2*row3)/3

tit_train = tit_data[1:int(mark1)][0:row1]
tit_test = tit_data[int(mark1):][0:row1]

#print(tit_data)
#print(tit_train)

win_train = win_data[0:int(mark2)][0:row2]
win_test = win_data[int(mark2):][0:row2]

#print(win_train)

wea_train = wea_data[0:int(mark3)][0:row3]
wea_test = wea_data[int(mark3):][0:row3]

#print(wea_test.head())

#print(mark1)

#print(tit_data[int(mark1):][0:row1])

#Classification using Random Forest
def Random_Forest():
    attributes = ["Sex","Age","Siblings/Spouses Aboard"]
    
    #print(attributes["Sex"])
    
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
    
#Random_Forest()

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

#Regression using Linear Regression
def Linear_Regression():
    attributes1 = ["Precip","MaxTemp","MinTemp","Snowfall","PRCP","MAX","MIN","SNF"]
    
    reg = linear_model.LinearRegression()
    reg.fit(wea_train[attributes1],wea_train.MeanTemp)
    
    #print(reg.coef_)
    #print(reg.intercept_)
    
    accuracy = cross_validation.cross_val_score(reg,wea_train[attributes1],wea_train.MeanTemp,cv=2)
    print(accuracy.mean())    
    
    print(reg.predict(wea_test[attributes1]))

#Linear_Regression()    

#Association Mining using Apriori Algorithm
def Apriori():
    transactions = []
    #print(row4)
    for i in range(0,row4):
        for j in range(0,col4):
            transactions.append(mar_data.values[i,j])
            
    rules = apriori(transactions,min_support=0.015,min_confidence=0.2,min_lift=3,min_length=2)
    
    result = list(rules)
    print(result)

#Apriori()






















