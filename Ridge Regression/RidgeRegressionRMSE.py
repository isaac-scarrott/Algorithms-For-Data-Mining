
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 22:24:08 2019

@author: isaacscarrott
"""
#Import the correct variables
from numpy import linalg, dot, identity, asarray, asmatrix, random, sqrt, mean
from pandas import read_csv
from matplotlib.pyplot import title, loglog, legend, xlabel, ylabel

#Where all of the weights are calculated using matrix multiplication and inversion and addition
def ridge_regression(features_train,  y_train, regularisationFactor):    
    
    #Identity Matrix in the shape of the features
    I = identity(features_train.shape[1])
    
    #This is the equasion to work out the weights of each of the twelve features using the equasion discussed in lecture 3 slide 63 and return the value of this as a numpy array
    parameters = linalg.solve(dot(features_train.T, features_train) + dot(regularisationFactor, I), dot(features_train.T, y_train))
    return asarray(parameters)

#This will calculate the RMSE for weights and features provided in comparison to predicted y
def eval_regression(parameters, features, y):
    #Calcaulates the predicted Y value
    featuresPlottingMultiplied = (dot(features,parameters)).reshape(len(features),)
    
    #Calculates the Root Mean Square Error and returns it
    rmse = sqrt(mean((featuresPlottingMultiplied-y)**2))
    print(rmse)
    return rmse

#Loads in the training datafram
featuresTrainDF = asarray(read_csv("Data/train.csv"))

#Creates two empty arrays for the values that the RMSE will return
rmseTrain = []
rmseTest = []

#Calulcates the number of features in dataframe
feature = list(range(3,len(featuresTrainDF.T)))

#Regularisation factors that we will loop through
values = [10**-6, 10**-4, 10**-2, 10**-1, 10**0, 10**1, 10**2, 10**3]

#Loops through so we can get an accurate RMSE
for x in range(10):
    
    #Shuffles the data randomly
    featuresTrain = asarray(sorted(featuresTrainDF, key=lambda k: random.random()))
    
    #Splits the data into approximatly 70% training and 30% test
    training, test = featuresTrain[:(int(len(featuresTrain)*0.7)),:], featuresTrain[(int(len(featuresTrain)*0.7)):,:]
    
    #Splits the columns of the test and training data up into the corrisponding variables
    xTrain = training[:,1]
    yTrain = training[:,2]
    xTest = test[:,1]
    yTest = test[:,2]
    featuresTrain = training[:,feature]
    featuresTest = test[:,feature]
    
    #Reinitalises the temp variables
    weights = []
    featuresMultiplied = []
    rmseTrain.append([])
    rmseTest.append([])
    
    #Loops through the regularisation factors
    for x in values:
        #Puts the returned array of weights for the given regularisation factor into the weight array 
        weights.append(ridge_regression(asmatrix(featuresTrain), asmatrix(yTrain).T, x))
        
        #Calulcates the RMSE of the regularisation factor and puts that list into the corrisponding list
        rmseTrain[-1].append(eval_regression(weights[-1], featuresTrain, yTrain))
        rmseTest[-1].append(eval_regression(weights[-1], featuresTest, yTest))

#Calulcates the main of each of the regularisation factors for test and train to get accurate results
rmseTrain = mean(rmseTrain, axis=0)
rmseTest = mean(rmseTest, axis=0)

#Plots the RMSE for the test and training data the same logarithmic plot
loglog(values, rmseTrain, label="Training Data")
loglog(values, rmseTest, label="Testing Data")
legend()
title("RMSE for λ 10**-6 - 10**3")
xlabel("λ")
ylabel("RMSE")
