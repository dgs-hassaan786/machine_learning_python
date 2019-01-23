#Data Preprocessing

#Importing the libraries
#we are using 3 libraries every time

#numpy includes all the mathematics
import numpy as np 

#matplotlib is used for ploting the charts 
import matplotlib.pyplot as plt

#pandas is used for import data set and manage them
import pandas as pd

# =============================================================================
# import sys
# print(sys.version)
# print(sys.executable)
# print(sys.path)
# =============================================================================

#setting the working directory folder 
#this can be done in panda
working_dir = 'F:/workspaces/machine_learning/Machine_Learning_AZ_Template_Folder/Part 1 - Data Preprocessing/Section 2 -------------------- Part 1 - Data Preprocessing --------------------/Data.csv'

dataset = pd.read_csv(working_dir)

#in machine learning you have to define the matrices
#there are independent and dependent matrices
#the dependent are combination of independent one

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values