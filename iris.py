
"""
Program: iris.py
Author:  Jorge Martin Joven
Date:    March 28,2020
Goal:    Based on the information provided by dataset (iris_dataset), we want to build a machine learning model
         able to predict the species of new samples using their new measurements. 
"""
from sklearn.datasets import load_iris #load the data
from sklearn.model_selection import train_test_split  #this function shuffles and split the data in the 'train set' and the 'test set'.
                                                      # it shuffles to make sure that both sets have all classes (labels).

iris_dataset = load_iris()

#List of the Fields of the dataset
print("Keys of the iris_dataset: \n{}".format(iris_dataset.keys()))
#DESCR is a short description of the dataset
print(iris_dataset['DESCR'][:193] + "\n...")
#Target_names is the list of iris species
print("Target names: {}".format(iris_dataset['target_names']))
#Feature_names is hte list of the parameters that have been measured. 
print("Feature names: {}".format(iris_dataset['feature_names']))
#The datatype is a numpy array. It contains the measurements of teh feature_names parameters
print("Type of data: {}".format(type(iris_dataset['data'])))
#Shapre of the data (150, 4). 150 samples (flowers measured) and 4 columns/features(4 types of measurements)
print("Shape of data: {}".format(iris_dataset['data'].shape))
#'target' is a 1D array containing the species for the 150 samples (coded as 0: setosa, 1: versicolor, and 2: virginica)
print("Target: \n{}".format(iris_dataset['target']))

#Usually, te data is splited in two parts: one is to build/train the ML model (75%) and the other one is test the model (25%)
#X_ are sets of data, whereas y_ are labels (categories). They are NumPy arrays
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))

### DATA EXPLORARTION: LOOKING AT THE DATA ############################################################################
#######################################################################################################################
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import mglearn
#Create dataframe form data in X_train
#label the columns using the strings in iris_dataset.feature_name
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
#Create a scatter matrix from the dataframe, color by y_train
grr = scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15), marker='o', hist_kwds={'bins':20},s=60, alpha=.8, cmap=mglearn.cm3) 

#plt.show()
#Judging by the scatterplots, we can see that the three classes seem to be well separated. 
#Therefore, a machine learning approach will probably be good.

### BUILDING THE MODEL: k-NEAREST NEIGHBOURS ##########################################################################
#######################################################################################################################

from sklearn.neighbors import KNeighborsClassifier

#Create an object 'knn' and we use only 1 neighbour (n_neighbors=1) to classify the new datapoint.
knn = KNeighborsClassifier(n_neighbors=1)

#Fit the model (to build the model) with X_train as data and y_train as training labels
print(knn.fit(X_train, y_train))

### MAKING PREDICTIONS ###############################################################################################
######################################################################################################################
import numpy as np 

# We first create a data point (a new sample) to be classified
X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new shape: {}".format(X_new.shape))
 
#To make the prediction we call teh 'predict' method

prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(iris_dataset['target_names'][prediction]))

### EVALUATING THE MODEL ############################################################################################
#####################################################################################################################

#We estimate ACCURACY. This is the fraction of flowers our model correctly asigns using our 'test sample'

y_pred = knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred)) #these are the predictions of our model

print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))

#Or we can use the 'score' method from the 'knn' object
#print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))