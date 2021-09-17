#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A case study of anomaly detection in Python
- 1st: create a dummy dataset
- 2nd: Detecting anomalies just by Visualization
- 3rd: Clustering based approach for anomaly detection
    k-means clustering is a method of vector quantization (vq), originally from 
    signal processing, that aims to partition n observations into k clusters 
    in which each observation belongs to the cluster with the nearest mean 
    (cluster centers or cluster centroid), serving as a prototype of the 
    cluster.
    This method looks at the data points in the set and groups those that are 
    similar (e.g. through Euclidean distance) into a predefined number K of 
    clusters. A threshold value can be added to detect anomalies: if the 
    distance between a data point and its nearest centroid is greater than
    the threshold value, then it is an anomaly. 
    
- 4th: Anomaly detection as a classification problem
    Proximity-based anomaly detection (by use of  k-NN classification method): 
        The basic idea here is that the proximity of an anomaly data point to 
        its nearest neighboring data points largely deviates from the proximity 
        of the data point to most of the other data points in the data set.

While detecting anomalies, we almost always consider ROC and Precision as it 
gives a much better idea about the model's performance. 
An ROC curve (receiver operating characteristic curve) is a graph showing the 
performance of a classification model at all classification thresholds. This 
curve plots two parameters: True Positive Rate. False Positive Rate.
Infact, The ROC curve shows the trade-off between sensitivity (or TPR) and 
specificity (1 – FPR). Classifiers that give curves closer to the top-left 
corner indicate a better performance. ... The closer the curve comes to the 
45-degree diagonal of the ROC space, the less accurate the test.

Precision talks about how precise/accurate your model is out of those predicted 
positive, how many of them are actual positive. Precision is a good measure to 
determine, when the costs of False Positive is high.
@author: Mahdieh khosravi
"""
# Import the necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#=============================================================================#
#                              Create a dummy dataset
#=============================================================================#
# Use a predefined style set
plt.style.use('ggplot')

# Import Faker: For generating the names (& make them look like the real ones) 
from faker import Faker

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

# Manually change the salary entries of two individuals
salary_df.at[16, 'Salary (in USD)'] = 23
salary_df.at[65, 'Salary (in USD)'] = 17

# Verify if the salaries were changed
print(salary_df.loc[16])
print(salary_df.loc[65])

#=============================================================================#
#                     Detecting anomalies just by Visualization
#=============================================================================#
# Generate a Boxplot
salary_df['Salary (in USD)'].plot(kind='box')
plt.show()

# Generate a Histogram plot
salary_df['Salary (in USD)'].plot(kind='hist')
plt.show()

# Minimum and maximum salaries
print('Minimum salary ' + str(salary_df['Salary (in USD)'].min()))
print('Maximum salary ' + str(salary_df['Salary (in USD)'].max()))

#=============================================================================#
#    Clustering based approach for anomaly detection (Unsupervised)
#=============================================================================#
# Convert the salary values to a numpy array
salary_raw = salary_df['Salary (in USD)'].values

# For compatibility with the SciPy implementation
salary_raw = salary_raw.reshape(-1, 1)
salary_raw = salary_raw.astype('float64')

# Import kmeans from SciPy
from scipy.cluster.vq import kmeans
from scipy.cluster.vq import vq   # vector quantization (vq)
# Specify the data and the number of clusters to kmeans()
centroids, avg_distance = kmeans(salary_raw, 4)

# Get the groups (clusters) and distances (vq: Assign codes from a code book to observations)
groups, cdist = vq(salary_raw, centroids)

# Plot the groups
plt.scatter(salary_raw, np.arange(0,100), c=groups)
plt.xlabel('Salaries in (USD)')
plt.ylabel('Indices')
plt.show()

#=============================================================================#
#      Anomaly detection as a classification problem (Supervised)
#=============================================================================#
# First assign all the instances to 
salary_df['class'] = 0

# Manually edit the labels for the anomalies
salary_df.at[16, 'class'] = 1
salary_df.at[65, 'class'] = 1

# Veirfy 
print(salary_df.loc[16])
print(salary_df.head())

# Proximity-based anomaly detection by use of  k-NN
# Importing KNN module from PyOD
from pyod.models.knn import KNN

# Segregate (separated) the salary values and the class labels 
X = salary_df['Salary (in USD)'].values.reshape(-1,1)
y = salary_df['class'].values

# Train kNN detector
clf = KNN(contamination=0.02, n_neighbors=5)
# contamination - the amount of anomalies in the data (in %) which for our case 
# is 2/100 = 0.02
# n_neighbors - No. of neighbors to consider for measuring the proximity
clf.fit(X)

# Get the prediction labels of the training data
y_train_pred = clf.labels_ 
    
# Outlier scores
y_train_scores = clf.decision_scores_

# Import the utility function for model evaluation
from pyod.utils import evaluate_print

# Evaluate on the training data
evaluate_print('KNN', y, y_train_scores)

### Test the model by generating a sample value
# A salary of $37 (an anomaly right?)
X_test = np.array([[37.]])
# Check what the model predicts on the given test data point
print(clf.predict(X_test))

# A salary of $1256
X_test_abnormal = np.array([[1256.]])

# Predict
print(clf.predict(X_test_abnormal))

