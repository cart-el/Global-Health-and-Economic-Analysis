# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 01:02:00 2023

@author: USER
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from sklearn.metrics import silhouette_score
import scipy.optimize as opt
import err_ranges as err

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

def clean_data(data, a, b):
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
    data.columns.values[1] = a
    data.columns.values[0] = b
    
    # print information about the dataframe
    #print(data.info())
    
    # drop rows with missing values
    data = data.dropna(axis=0)
    
    # return cleaned data
    return data

df_general = clean_data(data, "GDP/Capita", "Life Expectancy")
print(df_general)

#Extract the 10 countries with highest GDP/Capita
df_TopGDPc = GDP_pc.nlargest(10, '2020')
print(df_TopGDPc)

#Extract the 10 countries with lowest GDP/Capita
df_BtmGDPc = GDP_pc.nsmallest(10, '2020')
print(df_BtmGDPc)

#Define a function to plot a bar chart
def barplot(data, title, ylabel, xlabel, colour):
    """
    Creates a horizontal bar chart using the provided data, title, 
    y-axis label, and x-axis label.
    
    Parameters:
        data (pandas DataFrame): Data to be used for the chart. 
        The first column should be the y-axis labels and the second column 
        should be the x-axis values.
        title (str): Title of the chart.
        ylabel (str): Label for the y-axis.
        xlabel (str): Label for the x-axis.
        
    Returns:
        None
    """
    plt.barh(data.iloc[:,0], data.iloc[:,1], color=colour)
    plt.title(title, fontweight='bold')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig('Barplot.png')
    plt.show()

#Plot bottom 10 countries with GDP/capita
barplot(df_BtmGDPc, 'Countries With Lowest GDP/Capita (USD)', 
                 'Countries', 'GDP/Capita (USD)', 'green')

#Plot top 10 countries with GDP/capita
barplot(df_TopGDPc, 'Countries With Highest GDP/Capita (USD)', 
                 'Countries', 'GDP/Capita (USD)', 'green')

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
    x = data.iloc[:, x_col]
    y = data.iloc[:, y_col]
    
    # create scatter plot
    plt.scatter(x, y)
    
    # add title, legend, x-axis label, and y-axis label
    plt.title(title, fontweight='bold')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(legend, loc='lower right', borderaxespad=1)
    
    #save plot figure
    plt.savefig('General Scatter Plot')
    
    # show plot
    plt.tight_layout()
    plt.show()
    
#Activate scatter plot function
scatter_plot(df_general, 1, 0, "Scatter Plot of GDP/Capita vs Life Expectancy", 
             "GDP/Capita", "Life Expectancy", "Countries")

#Filter the dataframe to extract the desired countries
df1 = df_general[df_general.iloc[:,1] <= 11000] #countries below world average
print(df1['Life Expectancy'].mean())
df2 = df_general[(df_general.iloc[:,1] > 11000) & 
                 (df_general.iloc[:,1] < 40000)] #
print(df2)
df3 = df_general[df_general.iloc[:,1] >= 40000]
print(df3)

#Plot a scatter plot of countries with a gdp per capita 
#below the world avergae, i.e < 11,000
#Activate scatter plot function
scatter_plot(df1, 1, 0, 
             "Countries with GDP/Capita Below 11,000 (World Average)", 
             "GDP/Capita (USD)", "Life Expectancy (Years)", "Countries")

#Normalise the data
scaler = preprocessing.MinMaxScaler()
df1_norm = scaler.fit_transform(df1)
print(df1_norm)

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

#Activate the optimum k function to get the number of effective clusters
optimise_k_num(df1_norm, 10)


#Create a function to run the KMeans model on the dataset
def kmeans_model(data, n_clusters):
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
    clusters = kmeans.predict(data)
    centroids = kmeans.cluster_centers_
    inertia = kmeans.inertia_
    
    return clusters, centroids, inertia

#Activate KMeans clustering function
clusters, centroids, inertia = kmeans_model(df1, 3)
print("Clusters: ", clusters)
print("Centroids: ", centroids)
print("Inertia: ", inertia)

#Calculate the silhouette score for the number of clusters
sil_0 = silhouette_score(df1_norm, clusters)
print(sil_0)

df1['Clusters'] = clusters
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
    
    df.iloc[:,1]
    df.iloc[:,0]
    cent1 = centroids[:,1]
    cent2 = centroids[:,0]
    plt.scatter(df.iloc[cluster == 0, 1], df.iloc[cluster == 0, 0], s=50,
               c='blue', label='Cluster 0')
    plt.scatter(df.iloc[cluster == 1, 1], df.iloc[cluster == 1, 0], s=50,
               c='orange', label='Cluster 1')
    plt.scatter(df.iloc[cluster == 2, 1], df.iloc[cluster == 2, 0], s=50,
               c='green', label='Cluster 2')
    #Centroid plot
    plt.scatter(cent1, cent2, c='red', s=100, label='Centroid')
    plt.title('Cluster of Countries with GDP/Capita Below 11,000 (World Average)',
              fontweight='bold')
    plt.ylabel('Life Expectancy', fontsize=12)
    plt.xlabel('GDP/Capita', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Clusters.png")
    plt.show()

#Activate the function to plot the clusters    
plot_clusters(df1, clusters, centroids)

#Carry out cluster analysis by plotting a bar chart showing the country 
#distribution in each cluster
sns.countplot(x='Clusters', data=df1)
plt.savefig('Cluster distribution.png')
plt.title('Cluster Distribution of Countries with GDP/Capita Below'
    ' (World Average)', fontweight='bold')
plt.show()

#Define a polynomial function to plot a curve fit curve
def fit_polynomial(x, a, b, c, d):
    """
    Fit a polynomial of degree 3 to a given set of data points.
    
    Parameters: 
    x: x-coordinates of the data points.
    a,b,c,d: function coefficients.
    
    Returns: Optimal values for the coefficients of the polynomial.
    """
    #popt, pcov = curve_fit(fit_polynomial, x, y)
    return  a*x**3 + b*x**2 + c*x + d

#Initialise variables
x_axis = df1.values[:,1]
y_axis = df1.values[:,0]

#Instantiate the curvefit function
popt, pcov = opt.curve_fit(fit_polynomial, x_axis, y_axis)
a, b, c, d = popt
print('y = %.5f * x^3 + %.5f * x^2 + %.5f * x + %.5f' % (a, b, c, d))
#print(pcov)

#Generate the curvefit line variables
d_arr = df1.values[:,1] #convert data to an array
x_line = np.arange(min(d_arr), max(d_arr)+1, 1) #a random range of points
y_line = fit_polynomial(x_line, a, b, c, d) #generate y-axis variables 
plt.scatter(x_axis, y_axis, label="Countries") #scatterplot
#plot the curvefit line
plt.plot(x_line, y_line, '--', color='black', linewidth=3, label="Curvefit")
plt.title('Cluster of Countries showing Prediction Line (Curvefit)',
              fontweight='bold')
plt.ylabel('Life Expectancy (Years)', fontsize=12)
plt.xlabel('GDP/Capita (USD)', fontsize=12)
plt.legend(loc='lower right')
plt.annotate('y = 0.00671x + 58.308', (3000, 55), fontweight='bold')
plt.savefig("Scatterplot Prediction Line.png")
plt.show()

#Generate the confidence interval and error range
sigma = np.sqrt(np.diag(pcov))
low, up = err.err_ranges(d_arr, fit_polynomial, popt, sigma)
#print(low, up)
#print(pcov) 

ci = 1.95 * np.std(y_axis)/np.sqrt(len(x_axis))
lower = y_line - ci
upper = y_line + ci
print(f'Confidence Interval, ci = {ci}')

#plot showing best fitting function and the error range
plt.scatter(x_axis, y_axis, label="Countries")
plt.plot(x_line, y_line, '--', color='black', linewidth=3, 
         label="Curvefit")
plt.fill_between(x_line, lower, upper, alpha=0.3, color='green', 
                 label="Error range")
plt.title('Cluster Showing Prediction Line (Curvefit) & Error Range',
              fontweight='bold')
plt.ylabel('Life Expectancy (Years)', fontsize=12)
plt.xlabel('GDP/Capita (USD)', fontsize=12)
plt.annotate(f'C.I = {ci.round(3)}', (7800, 60), fontweight='bold')
plt.legend(loc='lower right')
plt.savefig("Scatterplot Prediction Line.png")
plt.show()