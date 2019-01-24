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

#we are dealing with the missing data
#for mean we are importing preprocessing from
#sklearn and using the Imputer Class
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values= np.NaN,strategy='mean')
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])


#Encoding the categorical data
from sklearn.preprocessing import LabelEncoder
#from sklearn.compose import ColumnTransformer

labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])

#we are dummy encoding as the machine learning algorithms will be
#confused with the values like Spain > Germany > France
from sklearn.preprocessing import OneHotEncoder



onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()



labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state= 0)
