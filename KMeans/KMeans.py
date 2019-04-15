#Note that sometimes the centroid are placed badly and the program crashes
#If this happens please restart the kernal and start again
#To fix this I would impliment a better way to place the centroids like

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 18:33:00 2019

@author: isaacscarrott
"""
#Imports that have be defined by the breif
from numpy import zeros ,random, asarray, average, append, where, var,isnan, sqrt
from pandas import read_csv
from matplotlib.pyplot import plot, figure, xlabel, ylabel, title

#The kMeansClass
class kMeansClass:
    # Initializer / Instance Attributes
    def __init__(self, k, dataset):
        #Used to store centroids in a numpy array arrays that will contain k number
        #of NumPy arrays containing zeros with the shape (k,4)
        self.centroids = zeros((k,4))
        #Used to store the value of k or numnber of clusters as an integer
        self.k = k
        #Used to store the dataset given in a numpy array with the relevant columns
        self.dataset = asarray(dataset[['stem_length','stem_diameter', 'leaf_length','leaf_width']])
        #Used to store the average distance away from the nearest centroid for every iteration
        self.SSE = []
        #Used to stroe the varience of the distance to the assigned centroid
        self.variance = -1.0
         
    #This function will initalise all of the centroids randomly on the first iteration
    def initialise_centroids(self):
        #Gets the upper bounds of each of the axis
        max_x = self.dataset[:,0].max()
        max_y = self.dataset[:,1].max()
        max_z = self.dataset[:,2].max()
        max_k = self.dataset[:,3].max()
        
        #Gets the lower bounds of each of the axis
        min_x = self.dataset[:,0].min()
        min_y = self.dataset[:,1].min()
        min_z = self.dataset[:,2].min()
        min_k = self.dataset[:,3].min()
        
        #Will initalise a 4d centroid point using the upper and lower bounds defined earlier for each value of K
        for x in range(self.k):
            self.centroids[x,0] = random.uniform(min_x,max_x)
            self.centroids[x,1] = random.uniform(min_y,max_y)
            self.centroids[x,2] = random.uniform(min_z,max_z)
            self.centroids[x,3] = random.uniform(min_k,max_k)
        

    #Used to calcuate the euclidean distance to each of the centroids using 2 4d arrays
    def compute_euclidean_distance(self, vec_1, vec_2):
        distance = 0
        for x in range(3):
            distance = distance + (vec_1[x] - vec_2[x])**2
        #Returns the euclidean distance to the selected centorid
        return distance
    
    #The main function for k means, this will reposition centroids and assign each row to a cluster
    def kmeans(self):
        
        #Checks if centroids exists if they don't this will run and initalise the centroids
        if len(self.dataset.T) != 5+self.k:
            self.initialise_centroids()
            #This will create k number of columns for the euclidean distance to each of the centroids and fill it with zeros
            for x in range(self.k+1):
                self.dataset = append(self.dataset, zeros((len(self.dataset), 1)), axis=1)
                
        #This will run if the centroids have already been initiated. It will recalculate the centroids based on the average coordiante of the cluster
        else: 
            for x in range(self.k):
                self.centroids[x, 0] = average(self.dataset[where(self.dataset[:,self.k+4]==x+1),0])
                self.centroids[x, 1] = average(self.dataset[where(self.dataset[:,self.k+4]==x+1),1])      
                self.centroids[x, 2] = average(self.dataset[where(self.dataset[:,self.k+4]==x+1),2])
                self.centroids[x, 3] = average(self.dataset[where(self.dataset[:,self.k+4]==x+1),3])
                
        #Loop through each row then centroid and calcualte the distance to each centroid
        for rindex, row in enumerate(self.dataset):
            smallest=0
            for cindex, centroid in enumerate(self.centroids):
                self.dataset[rindex,4+cindex] = self.compute_euclidean_distance([row[0], row[1],row[2], row[3]], centroid)
                #Checks if it the first loop of the centroids array or if the newly calculated euclidean distance is
                #smaller than the one that is stored in the temporary variable holder
                if self.dataset[rindex,4+cindex] < smallest or cindex == 0:
                    smallest = self.dataset[rindex,4+cindex]
                    self.dataset[rindex,self.k+4] = cindex+1
        
        #Creates an new item in the SSE with the sum of the euclidean distance to the first centroid where the cluster it is assigned to is 1
        #I have to sum it twice due to the shape of the numpy array being bad and having an array nestest in an array
        self.SSE.append(sum(sum((self.dataset[where(self.dataset[:,self.k+4]==1),4])**2)))
        
        #Appends the last item in the SSE list with the sum of the euclidean distance to centroid x+2 where the cluster it is assigned to is x+2
        for x in range(self.k-1):
            self.SSE[-1] = self.SSE[-1] + sum(sum((self.dataset[where(self.dataset[:,self.k+4]==x+2),x+5])**2))
            
        
        #If there are 5 items in the aeverage distance list it will calculate the varience in the last 3 clusters of the distance to the centroid 
        if len(self.SSE) > 4:
            self.variance = var([self.SSE[-1],self.SSE[-2],self.SSE[-3],self.SSE[-4],self.SSE[-5]])

        return self.dataset
    #Used to plot the graph of each object and the objective function
    def plot(self, figuresCreated):
        
        #Loops through each of the clusters
        for x in range(self.k):
            
            #Used to label the clusters so they can be easily identified
           tempLabel = 'C' + str(x+1)
           
           #creates a new figutre
           figure(figuresCreated)
           
           #Plots the 4d data on a 2d graph where the x axis is stem length and the y axis is stem diameter
           plot(self.dataset[where(self.dataset[:,self.k+4]==x+1),0], self.dataset[where(self.dataset[:,self.k+4]==x+1),1],'.', c=tempLabel)
           plot(self.centroids[:,0], self.centroids[:,1], 'kD')
           title("K Means " + str(self.k))
           xlabel("Stem Length")
           ylabel("Stem Diameter")
           
        #Creares a new figure
        figure(figuresCreated+1)
        #Loops through each of the clusters
        #Plots the objective function, where the x axis is the iteration and the y axis is the average distance away from the assignewd clusters centroids
        plot(self.SSE)
        xlabel("Iteration")
        ylabel("SSE")
        title("SSE for K Means " + str(self.k))
        return 2

#Reads in the dataframe
plantsDF = read_csv("plants.csv")

#Number of figures that have been created
figuresCreated = 1

#Creates two objects one for k=3 and one for k=4 using the dataframe
kmeans3 = kMeansClass(3, plantsDF)
kmeans4 = kMeansClass(4, plantsDF)

#Loops until the varience gets below a certain threshold (min number of iterations is 5)
while True:
    #checks if variance is not a number. usually if variance is nan then there are not k number of centroids.
    #If the variance is not a number then it will delete the object, create it again so the centroids are reintialised
    if isnan(kmeans3.variance):
        del kmeans3
        kmeans3 = kMeansClass(3, plantsDF)
        continue
    if isnan(kmeans4.variance):
        del kmeans4
        kmeans4 = kMeansClass(4, plantsDF)
        continue
    #If the varince of the SSE is above the chosen threshold then it
    #will continue to try and create better clusters, however if it encounters any errors that crash the algorithm
    #then it will delete the object, create it again so the centroids are reintialised
    try:
        if kmeans3.variance > 0.00000001 or (kmeans3.variance) == - 1:
            #Runs the kmeans functions for k means 3
            lol =kmeans3.kmeans()
    #if something like the centroids have been initiated badly then it
    #will just reinitalise the object
    except:
        del kmeans3
        kmeans3 = kMeansClass(3, plantsDF)
        continue
    #See the above try statement
    try:
        if (kmeans4.variance > 0.00000001 or (kmeans4.variance) == -1):
            kmeans4.kmeans()
    except:
        del kmeans4
        kmeans4 = kMeansClass(4, plantsDF)
        continue
    #If both of the variences for k=3 and k=4 are below the threshold and the variance value has been assigned
    #then the loop will stop as the objective function is at it's minimum value
    if kmeans3.variance < 0.00000001 and kmeans4.variance < 0.00000001 and kmeans3.variance >= 0 and kmeans4.variance >= 0:
        break
    
#Calls the function to plot the scatter graph of the clusters and the objective function. Also adds onto the images created
figuresCreated = figuresCreated + kmeans3.plot(figuresCreated) 
figuresCreated = figuresCreated + kmeans4.plot(figuresCreated)
