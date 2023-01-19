# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 11:15:38 2023

@author: USER
"""

#Import necessary libraries

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
    data_dir = "C:/Users/USER/Desktop/Data Science Course/Applied Data Science 1/ASS 3/"
    file = data_dir + data
    WB_data = pd.read_excel(file, skiprows=3)
    return WB_data, WB_data.transpose()
       
WB_data1 = tool("WD_data.xls")
print(WB_data1)

WB_data2 = tool("Life_Expectancy.xlsx")
WB_data2 = WB_data2[0]
Life_Exp = WB_data2.iloc[:, [0,64]]
print(Life_Exp)

WB_data3 = tool("GDP Per Capita.xlsx")
WB_data3 = WB_data3[0] 
GDP_pc = WB_data3.iloc[:, [0,64]]
GDP_pc = GDP_pc.round(3)
print(GDP_pc)

joint = pd.concat([Life_Exp, GDP_pc["2020"]], axis=1)
print(joint)

joint = joint.drop(index=[1,3,6,7,11,27,36,51,52,57,61,62,63,64,65,68,
                               73,74,78,84,91,95,98,102,103,104,105,107,108,
                               110,125,128,134,135,136,137,139,140,142,147,
                               149,153,155,156,161,164,170,184,181,183,188,
                               191,196,198,204,212,214,215,217,218,225,226,
                               228,230,231,236,238,240,241,249,255,259,261])
print(joint)


joint = joint.set_index('Country Name')
print(joint)

joint.columns.values[1] = "GDP/Capita"
joint.columns.values[0] = "Life Expectancy"
print(joint)

print(joint.info())

joint = joint.dropna(axis=0)
print(joint)

#plot the scatter plot
x = joint.iloc[:,1]
y = joint.iloc[:,0]
plt.scatter(x, y)

#divide the dataframe into subsets of interest
df1 = joint[joint.iloc[:,1] <= 11000]
print(df1)

df2 = joint[(joint.iloc[:,1] > 11000) & (joint.iloc[:,1] < 60000)]
print(df2)

df3 = joint[joint.iloc[:,1] >= 60000]
print(df3)

#Normalize the data
scaler = preprocessing.MinMaxScaler()
df1_norm = scaler.fit_transform(df1)
print(df1_norm)

#plot the normalised scatter plot
x1 = df1_norm[:,1]
y1 = df1_norm[:,0]
plt.scatter(x1, y1)

#Define the kMeans function
def optimise_k_num(data, max_k):
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
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.grid(True)
    plt.show()
#plot the elbow curve    
optimise_k_num(df1_norm, 10)

model = KMeans(n_clusters=3, init='k-means++', 
              max_iter=100, random_state=0)
model.fit(df1_norm)

centroids = model.cluster_centers_
print(centroids)

cluster = model.predict(df1_norm)
print(cluster)

sil_0 = silhouette_score(df1_norm, cluster)
print(sil_0)

df1['K-Means'] = model.labels_
print(df1)