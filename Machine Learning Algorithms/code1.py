#Importing required libraries
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

#Importing the data
datafile1 = 'titanic.csv'
datafile2 = 'Wine.csv'
datafile3 = 'Summary of Weather.csv'
datafile4 = 'marketbasket.csv'
datafile5 = 'Ads_CTR_Optimisation.csv'

#Reading the data in csv format
tit_data = pd.read_csv(datafile1)
win_data = pd.read_csv(datafile2)
wea_data = pd.read_csv(datafile3)
mar_data = pd.read_csv(datafile4)
ads_data = pd.read_csv(datafile5)

#Storing the number of rows and columns in a particular variable
row1,col1 = tit_data.shape
row2,col2 = win_data.shape
row3,col3 = wea_data.shape
row4,col4 = mar_data.shape
row5,col5 = ads_data.shape

#Changing the string into a numeric value for computation
m = {'male' : 1, 'female' : 0}
tit_data["Sex"] = tit_data.Sex.map(m)

#print(tit_data.head())
scale(win_data)

#Cleaning the weather data
wea_data = wea_data.replace('T',0)
wea_data = wea_data.replace('#VALUE!',0)
wea_data.fillna(0,inplace=True)
#print(np.sum(np.invert(np.isreal(wea_data[attributes1]))))

mark1 = (2*row1)/3
mark2 = (2*row2)/3
mark3 = (2*row3)/3

#Dividing titanic data into training and testing data
tit_train = tit_data[1:int(mark1)][0:row1]
tit_test = tit_data[int(mark1):][0:row1]

#print(tit_data)
#print(tit_train)

#Dividing wine data into training and testing data
win_train = win_data[0:int(mark2)][0:row2]
win_test = win_data[int(mark2):][0:row2]

#print(win_train)

#Dividing weather data into training and testing data
wea_train = wea_data[0:int(mark3)][0:row3]
wea_test = wea_data[int(mark3):][0:row3]

#print(wea_test.head())

#print(mark1)

#print(tit_data[int(mark1):][0:row1])

#Classification using Random Forest
def Random_Forest():
    #Selecting the attributes to be used to form a classification model    
    attributes = ["Sex","Age","Siblings/Spouses Aboard"]
    
    #print(attributes["Sex"])
    
    x,y = tit_train[attributes],tit_train.Survived
    
    #print(x)
    #print(y)
    
    #Creating decision trees witha maximum depth of 4
    random_model = RandomForestClassifier(max_depth=4)
    
    #Finding the accuracy of the model
    validate = cross_validation.cross_val_score(random_model,x,y,cv=5)    
    print(validate.mean())
    
    #Training the data
    random_model.fit(x,y)
    
    #Predicting values of testing data
    prediction = random_model.predict(tit_test[attributes])
    print(prediction)
    
#Random_Forest()

#Clustering using K-Means    
def K_Means():
    #Generating 3 clusters with 3 centroids
    cluster = KMeans(n_clusters=3)
    
    #Training the data
    cluster.fit(win_data)
    
    #Deciding colors to represent each cluster
    color = np.array(['green','red','blue','yellow'])
    
    #Creating a scatter plot
    plt.scatter(x=win_data.Alcohol,y=win_data.Color_Intensity,c=color[win_data.Customer_Segment])
    
    plt.title('K-Means Classificatipn')
    plt.ylabel('Color Intensity')
    plt.xlabel('Alcohol')

#K_Means()

#Regression using Linear Regression
def Linear_Regression():
    #Selecting the attributes to be used to form a linear model    
    attributes1 = ["Precip","MaxTemp","MinTemp","Snowfall","PRCP","MAX","MIN","SNF"]
    
    reg = linear_model.LinearRegression()
    
    #Finding regression coefficients of the training data
    reg.fit(wea_train[attributes1],wea_train.MeanTemp)
    
    #print(reg.coef_)
    #print(reg.intercept_)
    
    #Finding the accuracy of the linear model
    accuracy = cross_validation.cross_val_score(reg,wea_train[attributes1],wea_train.MeanTemp,cv=2)
    print(accuracy.mean())    
    
    #Predicting values of testing data
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
    
    #Repeat for each row
    for i in range(0,row5):
        ad = 0
        max_random_value = 0
        
        #Using Bayesian Control Rule for each column
        for j in range(0,col5):
            random_beta_value = random.betavariate(num_of_rewards_1[j] + 1,num_of_rewards_0[j] + 1)
            
            #Finding the maximum random value generated
            if random_beta_value > max_random_value:
                max_random_value = random_beta_value
                ad = j
        
        #Append each ad in the empty list
        ads_selected.append(ad)
        
        #Calculate the reward
        reward = ads_data.values[i,ad]
        
        #Increment the number of reward 1's and reward 0's
        if reward == 1:
            num_of_rewards_1[ad] = num_of_rewards_1[ad] + 1
        else:
            num_of_rewards_0[ad] = num_of_rewards_0[ad] + 1
        
        #Calculate the total number of rewards
        total_rewards = total_rewards + reward
    
    #print(total_rewards)
        
    #Visualizing the data
    plt.hist(ads_selected)
    plt.ylabel('Number Of Times Each Ad Was Selected')
    plt.xlabel('Ad Number')

#Thompson_Sampling()