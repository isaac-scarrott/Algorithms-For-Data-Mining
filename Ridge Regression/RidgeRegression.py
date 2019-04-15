
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 14:27:45 2019

@author: isaacscarrott
"""
#Import the correct variables
from numpy import linalg, array, dot, identity, asarray, asmatrix
from pandas import read_csv
from matplotlib.pyplot import plot, figure, xlim, ylim, title, legend, savefig

#Where all of the weights are calculated using matrix multiplication and inversion and addition
def ridge_regression(features_train,  y_train, regularisationFactor):    
    
    #Identity Matrix in the shape of the features
    I = identity(features_train.shape[1])
    print(y_train)
    #This is the equasion to work out the weights of each of the twelve features using the equasion discussed in lecture 3 slide 63 and return the value of this as a numpy array
    parameters = linalg.solve(dot(features_train.T, features_train) + dot(regularisationFactor, I), dot(features_train.T, y_train))
    return asarray(parameters)

#Reads the 2 CSV files into two variables in a panada dataframe
featuresTrainDF = read_csv("Data/train.csv")
plottingDF = read_csv("Data/plotting.csv")

#Selects the respective columns for the variable form the respective dataframe and converts it to a numpy array
xTrain = array(featuresTrainDF['x'])
yTrain = array(featuresTrainDF['y'])
featuresTrain = array(featuresTrainDF[["features0","features1","features2","features3","features4","features5","features6","features7","features8","features9","features10","features11"]])
xPlotting = array(plottingDF['x'])
featuresPlotting = array(plottingDF[["features0","features1","features2","features3","features4","features5","features6","features7","features8","features9","features10","features11"]]).T

#Used to store the weights of each feature and the features multiplied by the weight
weights = []
featuresPlottingMultiplied = []

#Regularisation factors that we will loop through 
values = [10**-6, 10**-4, 10**-2, 10**-1]

#Loops through the regularisation factors
for x in values:
    
    #Puts the returned array of weights for the given regularisation factor into the weight array 
    weights.append(ridge_regression(asmatrix(featuresTrain), asmatrix(yTrain).T, x))
    #Calculates the learned function using the weight just calulcated and store it in an array
    featuresPlottingMultiplied.append(dot(featuresPlotting.T,weights[-1]))

#Loops through each of the learned functions
for index,y in enumerate(featuresPlottingMultiplied):
    
    #Sets the limits of the x and y axis
    xlim([-5, 5])
    ylim([-1000, 1000])
    titleTemp = 'Graph for λ = ' + str(values[index])
    #Plots the learned function and the training points on the same graph and labels it
    title(titleTemp)
    plot(xTrain, yTrain, 'o', label="Training Points")
    plot(xPlotting, y, label="Learned Function")
    legend()
    savefig(titleTemp + ".png")
    figure()


