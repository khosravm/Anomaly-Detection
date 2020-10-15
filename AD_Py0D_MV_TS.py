
"""
Unsupervised multivariate anomaly detection
PyOD : A Python library for detecting anomalies

- Cluster-based Local Outlier Factor (CBLOF):
   It calculates the outlier score based on cluster-based local outlier factor. 
   An anomaly score is computed by the distance of each instance to its cluster 
   center multiplied by the instances belonging to its cluster.
   
- Histogram-based Outlier Detection (HBOS):
   It assumes the feature independence and calculates the degree of anomalies 
   by building histograms.
   
- Isolation Forest:
   Similar in principle to Random Forest, it isolates observations by randomly 
   selecting a feature and then randomly selecting a split value between the 
   maximum and minimum values of that selected feature.
   
- K - Nearest Neighbors (KNN):
   In this method, distance a data pointto its kth nearest neighbor could be 
   viewed as the outlier score.
@author: khosravm
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.ensemble import IsolationForest
from numpy import percentile

## Dataset and its distribution ===============================================
df = pd.read_excel("Superstore.xls")
# url: https://community.tableau.com/s/question/0D54T00000CWeX8SAL/sample-superstore-sales-excelxls
print(df.columns)

# Plot correlation chart
sns.regplot(x="Sales", y="Profit", data=df)
sns.despine();

# CBLOF =======================================================================
from pyod.models.cblof import CBLOF
X1 = df['Sales'].values.reshape(-1,1)
X2 = df['Profit'].values.reshape(-1,1)

X = np.concatenate((X1,X2),axis=1)

outliers_fraction = 0.01
xx , yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
clf = CBLOF(contamination=outliers_fraction,check_estimator=False, random_state=0)
clf.fit(X)
scores_pred = clf.decision_function(X) * -1
y_pred = clf.predict(X)
n_inliers = len(y_pred) - np.count_nonzero(y_pred)
n_outliers = np.count_nonzero(y_pred == 1)

plt.figure(figsize=(8, 8))

df1 = df
df1['outlier'] = y_pred.tolist()
    
# sales - inlier feature 1,  profit - inlier feature 2
inliers_sales = np.array(df1['Sales'][df1['outlier'] == 0]).reshape(-1,1)
inliers_profit = np.array(df1['Profit'][df1['outlier'] == 0]).reshape(-1,1)
    
# sales - outlier feature 1, profit - outlier feature 2
outliers_sales = df1['Sales'][df1['outlier'] == 1].values.reshape(-1,1)
outliers_profit = df1['Profit'][df1['outlier'] == 1].values.reshape(-1,1)
         
print('CBLOF_OUTLIERS:',n_outliers,'INLIERS:',n_inliers)

plt.figure(figsize=(8, 8))
threshold = percentile(scores_pred, 100 * outliers_fraction)        
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)  #levels=np.linspace(threshold, Z.min(),7),

a = plt.contour(xx, yy, Z, levels=[threshold],linewidths=2, colors='red')
plt.contourf(xx, yy, Z, levels=[threshold, Z.max()],colors='orange')
b = plt.scatter(inliers_sales, inliers_profit, c='white',s=20, edgecolor='k')
    
c = plt.scatter(outliers_sales, outliers_profit, c='black',s=20, edgecolor='k')
       
plt.axis('tight')   
plt.legend([a.collections[0], b,c], ['learned decision function', 'inliers','outliers'],
           prop=matplotlib.font_manager.FontProperties(size=20),loc='lower right')      
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.title('Cluster-based Local Outlier Factor (CBLOF)')
plt.show();
# HBOS  =======================================================================
from pyod.models.hbos import HBOS
outliers_fraction = 0.01
xx , yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
clf = HBOS(contamination=outliers_fraction)
clf.fit(X)
scores_pred = clf.decision_function(X) * -1
y_pred = clf.predict(X)
n_inliers = len(y_pred) - np.count_nonzero(y_pred)
n_outliers = np.count_nonzero(y_pred == 1)
plt.figure(figsize=(8, 8))
df1 = df
df1['outlier'] = y_pred.tolist()
    
# sales - inlier feature 1,  profit - inlier feature 2
inliers_sales = np.array(df1['Sales'][df1['outlier'] == 0]).reshape(-1,1)
inliers_profit = np.array(df1['Profit'][df1['outlier'] == 0]).reshape(-1,1)
    
# sales - outlier feature 1, profit - outlier feature 2
outliers_sales = df1['Sales'][df1['outlier'] == 1].values.reshape(-1,1)
outliers_profit = df1['Profit'][df1['outlier'] == 1].values.reshape(-1,1)
         
print('HBOS_OUTLIERS:',n_outliers,'INLIERS:',n_inliers)
#threshold = percentile(scores_pred, 100 * outliers_fraction)
#Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
#Z = Z.reshape(xx.shape)
#
#plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)   # levels=np.linspace(Z.min(), threshold, 7),
#a = plt.contour(xx, yy, Z, levels=[threshold],linewidths=2, colors='red')
#plt.contourf(xx, yy, Z, levels=[threshold, Z.max()],colors='orange')
#b = plt.scatter(inliers_sales, inliers_profit, c='white',s=20, edgecolor='k')
#    
#c = plt.scatter(outliers_sales, outliers_profit, c='black',s=20, edgecolor='k')
#       
#plt.axis('tight')      
#plt.legend([a.collections[0], b,c], ['learned decision function', 'inliers','outliers'],
#           prop=matplotlib.font_manager.FontProperties(size=20),loc='lower right')      
#plt.xlim((0, 1))
#plt.ylim((0, 1))
#plt.title('Histogram-base Outlier Detection (HBOS)')
#plt.show();

# IForest  ====================================================================
from pyod.models.iforest import IForest

outliers_fraction = 0.01
xx , yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
clf = IForest(contamination=outliers_fraction,random_state=0)
clf.fit(X)
scores_pred = clf.decision_function(X) * -1

y_pred = clf.predict(X)
n_inliers = len(y_pred) - np.count_nonzero(y_pred)
n_outliers = np.count_nonzero(y_pred == 1)
plt.figure(figsize=(8, 8))

df1 = df
df1['outlier'] = y_pred.tolist()
    
# sales - inlier feature 1,  profit - inlier feature 2
inliers_sales = np.array(df1['Sales'][df1['outlier'] == 0]).reshape(-1,1)
inliers_profit = np.array(df1['Profit'][df1['outlier'] == 0]).reshape(-1,1)
    
# sales - outlier feature 1, profit - outlier feature 2
outliers_sales = df1['Sales'][df1['outlier'] == 1].values.reshape(-1,1)
outliers_profit = df1['Profit'][df1['outlier'] == 1].values.reshape(-1,1)
         
print('IF_OUTLIERS: ',n_outliers,'INLIERS: ',n_inliers)

#threshold = percentile(scores_pred, 100 * outliers_fraction)
#Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
#Z = Z.reshape(xx.shape)
#plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),cmap=plt.cm.Blues_r)
#a = plt.contour(xx, yy, Z, levels=[threshold],linewidths=2, colors='red')
#plt.contourf(xx, yy, Z, levels=[threshold, Z.max()],colors='orange')
#b = plt.scatter(inliers_sales, inliers_profit, c='white',s=20, edgecolor='k')
#    
#c = plt.scatter(outliers_sales, outliers_profit, c='black',s=20, edgecolor='k')
#       
#plt.axis('tight')
#plt.legend([a.collections[0], b,c], ['learned decision function', 'inliers','outliers'],
#           prop=matplotlib.font_manager.FontProperties(size=20),loc='lower right')      
#plt.xlim((0, 1))
#plt.ylim((0, 1))
#plt.title('Isolation Forest')
#plt.show();

# KNN =========================================================================
from pyod.models.knn import KNN
outliers_fraction = 0.01
xx , yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
clf = KNN(contamination=outliers_fraction)
clf.fit(X)
scores_pred = clf.decision_function(X) * -1
y_pred = clf.predict(X)
n_inliers = len(y_pred) - np.count_nonzero(y_pred)
n_outliers = np.count_nonzero(y_pred == 1)
plt.figure(figsize=(8, 8))

df1 = df
df1['outlier'] = y_pred.tolist()
    
# sales - inlier feature 1,  profit - inlier feature 2
inliers_sales = np.array(df1['Sales'][df1['outlier'] == 0]).reshape(-1,1)
inliers_profit = np.array(df1['Profit'][df1['outlier'] == 0]).reshape(-1,1)
    
# sales - outlier feature 1, profit - outlier feature 2
outliers_sales = df1['Sales'][df1['outlier'] == 1].values.reshape(-1,1)
outliers_profit = df1['Profit'][df1['outlier'] == 1].values.reshape(-1,1)
         
print('KNN_OUTLIERS: ',n_outliers,'INLIERS: ',n_inliers)

#threshold = percentile(scores_pred, 100 * outliers_fraction)
#
#Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
#Z = Z.reshape(xx.shape)
#
#plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),cmap=plt.cm.Blues_r)
#a = plt.contour(xx, yy, Z, levels=[threshold],linewidths=2, colors='red')
#plt.contourf(xx, yy, Z, levels=[threshold, Z.max()],colors='orange')
#b = plt.scatter(inliers_sales, inliers_profit, c='white',s=20, edgecolor='k')    
#c = plt.scatter(outliers_sales, outliers_profit, c='black',s=20, edgecolor='k')       
#plt.axis('tight')  
#plt.legend([a.collections[0], b,c], ['learned decision function', 'inliers','outliers'],
#           prop=matplotlib.font_manager.FontProperties(size=20),loc='lower right')      
#plt.xlim((0, 1))
#plt.ylim((0, 1))
#plt.title('K Nearest Neighbors (KNN)')
#plt.show();