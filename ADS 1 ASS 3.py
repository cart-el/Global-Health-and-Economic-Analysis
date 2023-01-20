# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 01:02:00 2023

@author: USER
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from sklearn.metrics import silhouette_score
import scipy.optimize as opt
#import err_ranges as err

#Define a function to read in the file to be processed
def tool (data):
    """
    Reads a file from the specified directory in world bank format and 
    outputs a dataframe 
    data: name of file in directory
   
    Returns
    -------
     Dataframe

    """
    data_dir = "C:/Users/USER/Desktop/Data Science Course/Applied Data Science 1/ASS 3/"
    file = data_dir + data
    WB_data = pd.read_excel(file, skiprows=3)
    return WB_data, WB_data.transpose()
       
#read files using function and extract the desired columns
# WB_data1 = tool("WD_data.xls") 
# print(WB_data1)

WB_data2 = tool("Life_Expectancy.xlsx")
#print(WB_data2)
WB_data2 = WB_data2[0]
#print(WB_data2)
Life_Exp = WB_data2.iloc[:, [0,64]]
#print(Life_Exp)
WB_data3 = tool("GDP Per Capita.xlsx")
WB_data3 = WB_data3[0] 
GDP_pc = WB_data3.iloc[:, [0,64]]
GDP_pc = GDP_pc.round(3)
#print(GDP_pc)

#Combine the dataframes into one
data = pd.concat([Life_Exp, GDP_pc["2020"]], axis=1)
#print(Data)
#Drop unwanted columns
data = data.drop(index=[1,3,6,7,11,27,36,51,52,57,61,62,63,64,65,68,
                                73,74,78,84,91,95,98,102,103,104,105,107,108,
                                110,125,128,134,135,136,137,139,140,142,147,
                                149,153,155,156,161,164,170,184,181,183,188,
                                191,196,198,204,212,214,215,217,218,225,226,
                                228,230,231,236,238,240,241,249,255,259,261])
# print(data)

def clean_data(data):
    """
    This function takes a dataframe as input and performs several 
    cleaning tasks on it:
    1. Sets the index to 'Country Name'
    2. Renames the columns to 'Life Expectancy' and 'GDP/Capita'
    3. Prints information about the dataframe
    4. Drops rows with missing values
    
    Parameters:
    data (pandas dataframe): the dataframe to be cleaned
    
    Returns:
    pandas dataframe: the cleaned dataframe
    """
    # set index to 'Country Name'
    data = data.set_index('Country Name')
    
    # rename columns
    data.columns.values[1] = "GDP/Capita"
    data.columns.values[0] = "Life Expectancy"
    
    # print information about the dataframe
    #print(data.info())
    
    # drop rows with missing values
    data = data.dropna(axis=0)
    
    # return cleaned data
    return data

df_general = clean_data(data)
print(df_general)

#Define functiong to normalize data
def normalize_df(df):
    """
    Normalize a given DataFrame using MinMaxScaler.
    
    Parameters:
    df: DataFrame to be normalized.
    :return: Normalized DataFrame.
    """
    scaler = preprocessing.MinMaxScaler()
    df_norm = scaler.fit_transform(df)
    return df_norm

#Activate scatter plot function
df_norm = normalize_df(df_general)
print(df_norm)

#Define function to plot a scatter plot
def scatter_plot(data, x_col, y_col, title, xlabel, ylabel, legend):
    """
    This function takes a dataframe and plots a scatter plot with the 
    specified columns for x and y values, as well as a title, x-axis label, 
    and y-axis label.
    
    Parameters:
    data (pandas dataframe): the dataframe to be plotted
    x_col (int or string):  index of x-axis column. 
    y_col (int or string):  index of y-axis column. 
    title (string): the title of the plot
    xlabel (string): the label for the x-axis
    ylabel (string): the label for the y-axis
    """
    # extract x and y values from the dataframe
    x = data[:, x_col]
    y = data[:, y_col]
    
    # create scatter plot
    plt.scatter(x, y)
    
    # add title, legend, x-axis label, and y-axis label
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(legend, loc='lower right', borderaxespad=1)
    
    #save plot figure
    plt.savefig('General Scatter Plot')
    
    # show plot
    plt.tight_layout()
    plt.show()
    
#Activate scatter plot function
scatter_plot(df_norm, 1, 0, "Scatter Plot of GDP/Capita vs Life Expectancy", 
             "GDP/Capita", "Life Expectancy", "Countries")
    
#Filter the dataframe to extract the desired countries
df1 = df_general[df_general.iloc[:,1] <= 11000] #countries below world average
print(df1)
df2 = df_general[(df_general.iloc[:,1] > 11000) & 
                 (df_general.iloc[:,1] < 40000)] #
print(df2)
df3 = df_general[df_general.iloc[:,1] >= 40000]
print(df3)

#Plot a scatter plot of countries with a gdp per capita 
#below the world avergae, i.e < 11,000
#Normalise the data
df1_norm = normalize_df(df1)
print(df1_norm)

#Activate scatter plot function
scatter_plot(df1_norm, 1, 0, 
             "Scatter Plot for Countries with GDP/Capita Below World Average", 
             "GDP/Capita", "Life Expectancy", "Countries")

#Define a function to determine the number of effective clusters for KMeans
def optimise_k_num(data, max_k):
    """
    A function to determine the optimal number of clusters for k-means 
    clustering on a given dataset. The function plots the relationship between 
    the number of clusters and the inertia, and displays the plot.
    
    Parameters:
    - data (array-like): the dataset to be used for clustering
    - max_k (int): the maximum number of clusters to test for
    
    Returns: None
    """
        
    means = []
    inertias = []
    
    for k in range(1, max_k):
        kmeans = KMeans(n_clusters=k, init='k-means++',
                        max_iter=1000)
        kmeans.fit(data)
        
        means.append(k)
        inertias.append(kmeans.inertia_)
        
    #generate the elbow_plot
    fig = plt.subplots(figsize=(8,5))
    plt.plot(means, inertias, "o-")
    plt.xlabel("Number of Clusters", fontsize=12)
    plt.ylabel("Inertia", fontsize=12)
    plt.title("Elbow Method Showing Optimal Number of K", fontsize=16, 
              fontweight='bold')
    plt.grid(True)
    plt.savefig("K-Means Elbow Plot.png")
    plt.show()
    
    return fig
#
optimise_k_num(df1_norm, 10)


#Create a function to run the KMeans model on the dataset
def kmeans_clustering(data, n_clusters):
    """
    Applies K-Means clustering on the data and returns the cluster labels.

    Parameters:
        data (numpy array or pandas dataframe) : The data to be clustered
        n_clusters (int) : The number of clusters to form.

    Returns:
        numpy array : The cluster labels for each data point
        numpy array : The cluster centers
        float : The inertia of the model
    """
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, 
                    random_state=0)
    kmeans.fit(data)
    labels = kmeans.predict(data)
    centroids = kmeans.cluster_centers_
    inertia = kmeans.inertia_
    
    return labels, centroids, inertia

#Activate KMeans clustering function
labels, centroids, inertia = kmeans_clustering(df1_norm, 3)
print("Labels: ", labels)
print("Centroids: ", centroids)
print("Inertia: ", inertia)

df1['Clusters'] = labels
print(df1)

#Define a function that plots the clusters
def plot_clusters(df, cluster, centroids):
    """
    Plot the clusters formed from a clustering algorithm.
    
    Parameters:
    df: DataFrame containing the data that was clustered.
    cluster: Array or Series containing the cluster labels for each 
    point in the data.
    centroids: Array or DataFrame containing the coordinates of the 
    cluster centroids.
    """
    df[:,1]
    df[:,0]
    cent1 = centroids[:,1]
    cent2 = centroids[:,0]
    plt.scatter(df[cluster == 0, 1], df[cluster == 0, 0], s=50,
               c='blue', label='Cluster 0')
    plt.scatter(df[cluster == 1, 1], df[cluster == 1, 0], s=50,
               c='orange', label='Cluster 1')
    plt.scatter(df[cluster == 2, 1], df[cluster == 2, 0], s=50,
               c='green', label='Cluster 2')
    #Centroid plot
    plt.scatter(cent1, cent2, c='red', s=100, label='Centroid')
    plt.title('Clustering of Countires with GDP/Capita Below World Average')
    plt.ylabel('Life Expectancy', fontsize=12)
    plt.xlabel('GDP/Capita', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.show()

#Activate the function to plot the clusters    
plot_clusters(df1_norm, labels, centroids)

#Define a polynomial function to plot a curve fit curve
def fit_polynomial(x, y):
    """
    Fit a polynomial of degree 3 to a given set of data points.
    
    Parameters: 
    x: x-coordinates of the data points.
    y: y-coordinates of the data points.
    
    Returns: Optimal values for the coefficients of the polynomial.
    """
    popt, pcov = curve_fit(fit_polynomial, x, y)
    return popt

x_axis = df1.values[:,1]
y_axis = df1.values[:,0]

popt, pcov = opt.curve_fit(fit_polynomial, x_axis, y_axis)
a, b, c, d = popt
print('y = %.5f * x^3 + %.5f * x^2 + %.5f * x + %.5f' % (a, b, c, d))

dec = df1.values[:,1]
x_line = np.arange(min(dec), max(dec)+1, 1)
y_line = fit_polynomial(x_line, a, b, c, d)

plt.scatter(x_axis, y_axis)
plt.plot(x_line, y_line, '--', color='black')
plt.show()