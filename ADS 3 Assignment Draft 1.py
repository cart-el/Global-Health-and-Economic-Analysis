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