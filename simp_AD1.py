#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 18:50:08 2020
https://blog.floydhub.com/introduction-to-anomaly-detection-in-python/
Here we have look in 3 different simple methods for detecting anomaly in a 
self-created dummy dataset:
    1- Detecting anomalies just by seeing
    2- Clustering based approach
    3- Classification based approach
@author: khosravm
"""

# Import the necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Comment out the following line if you are using Jupyter Notebook
# %matplotlib inline
# Use a predefined style set
plt.style.use('ggplot')

# Import Faker
from faker import Faker
fake = Faker()

# To ensure the results are reproducible
Faker.seed(4321)

names_list = []

fake = Faker()
for _ in range(100):
  names_list.append(fake.name())

# To ensure the results are reproducible
np.random.seed(7)

salaries = []
for _ in range(100):
    salary = np.random.randint(1000,2500)
    salaries.append(salary)

# Create pandas DataFrame
salary_df = pd.DataFrame(
    {'Person': names_list,
     'Salary (in USD)': salaries
    })

# Print a subsection of the DataFrame
print(salary_df.head())

# manually change the salary entries of two individuals
salary_df.at[16, 'Salary (in USD)'] = 23
salary_df.at[65, 'Salary (in USD)'] = 17

# Verify if the salaries were changed
print(salary_df.loc[16])
print(salary_df.loc[65])
#==============================================================================
## Seeing is believing: Detecting anomalies just by seeing
#==============================================================================
# Generate a Boxplot: minimum value, maximum value, 1st quartile values etc.
salary_df['Salary (in USD)'].plot(kind='box')
plt.show()

# Generate a Histogram plot
salary_df['Salary (in USD)'].plot(kind='hist')
plt.show()

# Minimum and maximum: an immediate way to confirm that the dataset contains anomalies
print('Minimum salary ' + str(salary_df['Salary (in USD)'].min()))
print('Maximum salary ' + str(salary_df['Salary (in USD)'].max()))

#==============================================================================
# Clustering based approach for anomaly detection
#==============================================================================
# Convert the salary values to a numpy array
salary_raw = salary_df['Salary (in USD)'].values

# For compatibility with the SciPy implementation
salary_raw = salary_raw.reshape(-1, 1)
salary_raw = salary_raw.astype('float64')

# Import kmeans from SciPy
from scipy.cluster.vq import kmeans,vq
    
# Specify the data and the number of clusters to kmeans()
centroids, avg_distance = kmeans(salary_raw, 4)

# Get the groups (clusters) and distances
groups, cdist = vq(salary_raw, centroids)

# plot the groups we have got
plt.scatter(salary_raw, np.arange(0,100), c=groups)
plt.xlabel('Salaries in (USD)')
plt.ylabel('Indices')
plt.show()
#==============================================================================
# Anomaly detection as a classification problem
#==============================================================================
"""
proximity-based anomaly detection:
    The basic idea is that the proximity of an anomaly data point to its nearest 
    neighboring data points largely deviates from the proximity of the data 
    point to most of the other data points in the data set.
    
    KNN(contamination, n_neighbors)
    contamination: amount of anomalies in the data (in percentage)
    n_neighbors: number of neighbors to consider for measuring the proximity
"""
# First assign all the instances to 
salary_df['class'] = 0

# Manually edit the labels for the anomalies
salary_df.at[16, 'class'] = 1
salary_df.at[65, 'class'] = 1

# Veirfy 
print(salary_df.loc[16])

# Importing KNN module from PyOD  
# PyOD is a Python library specifically developed for AD purposes.
from pyod.models.knn import KNN

# Segregate the salary values and the class labels 
X = salary_df['Salary (in USD)'].values.reshape(-1,1)
y = salary_df['class'].values

# Train kNN detector
clf = KNN(contamination=0.02, n_neighbors=5)
clf.fit(X)

# Get the prediction labels of the training data
y_train_pred = clf.labels_ 
    
# Outlier scores
y_train_scores = clf.decision_scores_

# Import the utility function for model evaluation
from pyod.utils import evaluate_print

# Evaluate on the training data
evaluate_print('KNN', y, y_train_scores)

# Test
# A salary of $37 (an anomaly right?)
X_test = np.array([[37.]])
# Check what the model predicts on the given test data point
clf.predict(X_test)
# A salary of $1256
X_test_abnormal = np.array([[1256.]])

# Predict
clf.predict(X_test_abnormal)