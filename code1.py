# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 14:29:48 2018

@author: lenovo pc
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
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
datafile5 = 'Ads_CTR_Optimisation.csv'

tit_data = pd.read_csv(datafile1)
win_data = pd.read_csv(datafile2)
wea_data = pd.read_csv(datafile3)
mar_data = pd.read_csv(datafile4)
ads_data = pd.read_csv(datafile5)

row1,col1 = tit_data.shape
row2,col2 = win_data.shape
row3,col3 = wea_data.shape
row4,col4 = mar_data.shape
row5,col5 = ads_data.shape

m = {'male' : 1, 'female' : 0}
tit_data["Sex"] = tit_data.Sex.map(m)

#print(tit_data.head())
scale(win_data)

#print(type(wea_data))
wea_data = wea_data.replace('T',0)
wea_data = wea_data.replace('#VALUE!',0)
wea_data.fillna(0,inplace=True)
#print(np.sum(np.invert(np.isreal(wea_data[attributes1]))))

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

#Association Mining using Apriori
def Apriori():
    transactions = []
    #print(row4)
    for i in range(0,row4):
        transactions.append([str(mar_data.values[i,j]) for j in range(0,col4)])
            
    #rules = apriori(transactions,min_support=0.015,min_confidence=0.2,min_lift=3,min_length=2)
    rules = apriori(transactions,min_support=0.015,min_confidence=0.4,min_length=2)
        
    result = list(rules)
    print(result)

#Apriori()

#Reinforcement Learning using Thompson Sampling
def Thompson_Sampling():
    ads_selected = []
    num_of_rewards_0 = [0] * col5
    num_of_rewards_1 = [0] * col5
    total_rewards = 0
    
    for i in range(0,row5):
        ad = 0
        max_random_value = 0
        
        for j in range(0,col5):
            random_beta_value = random.betavariate(num_of_rewards_1[j] + 1,num_of_rewards_0[j] + 1)
            
            if random_beta_value > max_random_value:
                max_random_value = random_beta_value
                ad = j
            
        ads_selected.append(ad)
        reward = ads_data.values[i,ad]
        
        if reward == 1:
            num_of_rewards_1[ad] = num_of_rewards_1[ad] + 1
        else:
            num_of_rewards_0[ad] = num_of_rewards_0[ad] + 1
        
        total_rewards = total_rewards + reward
    
    #print(total_rewards)
        
    plt.hist(ads_selected)
    plt.ylabel('Number Of Times Each Ad Was Selected')
    plt.xlabel('Ad Number')

#Thompson_Sampling()