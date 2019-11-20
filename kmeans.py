# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 20:20:07 2019

@author: lfxci
"""

##########################################################################################################################
##########################################################################################################################
# 1953
##########################################################################################################################

#K-Means clustering implementation
from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

fig = plt.gcf()
fig.set_size_inches(30, 30)
#plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

# ====
# Define a function that reads data in from the csv files  HINT: http://docs.python.org/2/library/csv.html
#data = pd.read_csv('data2008.csv')
data = pd.read_csv('data1953.csv')
#data = pd.read_csv('dataBoth.csv')
data_set_complete = data
data.head()

# Getting the values and plotting it
X = data.iloc[:, [1,2]].values 

from sklearn.cluster import KMeans
colors = ['r', 'g', 'b', 'y', 'c', 'm']
wcss =[]

k = int(input("Please enter number of Clusters: "))
iterations = int(input("Please enter number of Iterations: "))

#k = 2
#iterations = 10

# setup kmeans and fit clusters
kmeans = KMeans(n_clusters = k, init = 'k-means++', max_iter = iterations)
y_kmeans = kmeans.fit_predict(X)

# list of countries foreach cluster
# add clusters as key
data['Clusters'] = kmeans.labels_
data_set_complete['1953'] = kmeans.labels_

# set styles and formats
font = {'family' : 'sans-serif',
        'weight' : 'bold',
        'size'   : 5.5}

plt.rc('font', **font)

plt.subplot(3,1,1)

# foreach cluster, scatter man!
for i in range(0, k):     
    # mean life expectancy and birth rate for current cluster
    x_mean = np.mean(X[y_kmeans == i, 0])
    y_mean = np.mean(X[y_kmeans == i, 1])
    
    print('\n' + '1953 Countries for Cluster ' + str(i + 1))      
        
    # number of countries belonging to each cluster
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s = 10, c = colors[i], label = "Cluster " + str(i + 1) +  "\nNumber of Countries: " + str(np.array(np.where(data['Clusters'].values == i)[0]).size) + 
                "\nMean Birth " + str(round(x_mean, 2)) + "\nMean Life " + str(round(y_mean, 2)) + "\n")
    
    # display countries for current cluster
    countries = []
    countries_indexes = np.where(data['Clusters'] == i)
    for c in countries_indexes:
        countries = data.iloc[c]        
        
        # add country names to scatter points
        for i, country in enumerate(countries.values):
            plt.annotate(country[0],  (country[1], country[2]))
            print(country[0])
            
# scatter centroids and add legend
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 30, c = 'yellow', label = 'Centroids')
plt.title('1953 Clusters of Countries')
plt.xlabel('Birth rate')
plt.ylabel('Life expectency')
plt.legend()
plt.show()


##########################################################################################################################
##########################################################################################################################
# 2008
##########################################################################################################################

#K-Means clustering implementation
from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

fig = plt.gcf()
fig.set_size_inches(30, 30)
#plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

# ====
# Define a function that reads data in from the csv files  HINT: http://docs.python.org/2/library/csv.html
data = pd.read_csv('data2008.csv')
#data = pd.read_csv('data1953.csv')
#data = pd.read_csv('dataBoth.csv')
data.head()

# Getting the values and plotting it
X = data.iloc[:, [1,2]].values 

from sklearn.cluster import KMeans
colors = ['r', 'g', 'b', 'y', 'c', 'm']
wcss =[]

#k = int(input("Please enter number of Clusters: "))
#iterations = int(input("Please enter number of Iterations: "))

#k = 2
#iterations = 10

# setup kmeans and fit clusters
kmeans = KMeans(n_clusters = k, init = 'k-means++', max_iter = iterations)
y_kmeans = kmeans.fit_predict(X)

# list of countries foreach cluster
# add clusters as key
data['Clusters'] = kmeans.labels_
data_set_complete['2008'] = kmeans.labels_
font = {'family' : 'sans-serif',
        'weight' : 'bold',
        'size'   : 5.5}

plt.rc('font', **font)

plt.subplot(3,1,2)

# foreach cluster, scatter man!
for i in range(0, k):     
    # mean life expectancy and birth rate for current cluster
    x_mean = np.mean(X[y_kmeans == i, 0])
    y_mean = np.mean(X[y_kmeans == i, 1])
    
    print('\n' + '2008 Countries for Cluster ' + str(i + 1))      
        
    # number of countries belonging to each cluster
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s = 10, c = colors[i], label = "Cluster " + str(i + 1) +  "\nNumber of Countries: " + str(np.array(np.where(data['Clusters'].values == i)[0]).size) + 
                "\nMean Birth " + str(round(x_mean, 2)) + "\nMean Life " + str(round(y_mean, 2)) + "\n")
    
    # display countries for current cluster
    countries = []
    countries_indexes = np.where(data['Clusters'] == i)
    for c in countries_indexes:
        countries = data.iloc[c]        
        
        # add country names to scatter points
        for i, country in enumerate(countries.values):
            plt.annotate(country[0],  (country[1], country[2]))
            print(country[0])
            
# scatter centroids and add legend
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 30, c = 'yellow', label = 'Centroids')
plt.title('2008 Clusters of Countries')
plt.xlabel('Birth rate')
plt.ylabel('Life expectency')
plt.legend()
plt.show()



##########################################################################################################################
##########################################################################################################################
# Both
##########################################################################################################################

#K-Means clustering implementation
from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

fig = plt.gcf()
fig.set_size_inches(30, 30)
#plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

# ====
# Define a function that reads data in from the csv files  HINT: http://docs.python.org/2/library/csv.html
#data = pd.read_csv('data2008.csv')
#data = pd.read_csv('data1953.csv')
data = pd.read_csv('dataBoth.csv')
data.head()

# Getting the values and plotting it
X = data.iloc[:, [1,2]].values 

from sklearn.cluster import KMeans
colors = ['r', 'g', 'b', 'y', 'c', 'm']
wcss =[]

#k = int(input("Please enter number of Clusters: "))
#iterations = int(input("Please enter number of Iterations: "))

k = 4
#iterations = 10

# setup kmeans and fit clusters
kmeans = KMeans(n_clusters = k, init = 'k-means++', max_iter = iterations)
y_kmeans = kmeans.fit_predict(X)

# list of countries foreach cluster
# add clusters as key
data['Clusters'] = kmeans.labels_
font = {'family' : 'sans-serif',
        'weight' : 'bold',
        'size'   : 5.5}

plt.rc('font', **font)

plt.subplot(3,1,3)

# foreach cluster, scatter man!
for i in range(0, k):     
    # mean life expectancy and birth rate for current cluster
    x_mean = np.mean(X[y_kmeans == i, 0])
    y_mean = np.mean(X[y_kmeans == i, 1])
    
    print('\n' + 'Both Countries for Cluster ' + str(i + 1))      
        
    # number of countries belonging to each cluster
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s = 10, c = colors[i], label = "Cluster " + str(i + 1) +  "\nNumber of Countries: " + str(np.array(np.where(data['Clusters'].values == i)[0]).size) + 
                "\nMean Birth " + str(round(x_mean, 2)) + "\nMean Life " + str(round(y_mean, 2)) + "\n")
    
    # display countries for current cluster
    countries = []
    countries_indexes = np.where(data['Clusters'] == i)
    for c in countries_indexes:
        countries = data.iloc[c]        
        
        # add country names to scatter points
        for i, country in enumerate(countries.values):
            plt.annotate(country[0],  (country[1], country[2]))
            print(country[0])
            
# scatter centroids and add legend
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 30, c = 'yellow', label = 'Centroids')
plt.title('Both Clusters of Countries')
plt.xlabel('Birth rate')
plt.ylabel('Life expectency')
plt.legend()
plt.show()

print(data_set_complete)
