"""
A simple example of Univariate anomaly detection with IF method
- IsolationForest algorithm (a tree-based model) which is based on the fact 
that anomalies are data points that are few and different. 
outlier regions correspond to low probability areas.
@author: khosravm
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.ensemble import IsolationForest

## Dataset and its distribution ===============================================
df = pd.read_excel("Superstore.xls")
# url: https://community.tableau.com/s/question/0D54T00000CWeX8SAL/sample-superstore-sales-excelxls
print(df.columns)


# Plot dataset: Sales Column
print(df['Sales'].describe())
plt.subplot(411)
plt.scatter(range(df.shape[0]), np.sort(df['Sales'].values))
plt.xlabel('index')
plt.ylabel('Sales')
plt.title("Sales distribution")
sns.despine()   # Remove the top and right spines from plot(s)

plt.subplot(412)
sns.distplot(df['Sales'])
plt.title("Distribution of Sales")
sns.despine()

print("Skewness: %f" % df['Sales'].skew()) # a measure of the asymmetry of the probability distribution
print("Kurtosis: %f" % df['Sales'].kurt()) # the sharpness of the peak of a frequency-distribution curve
#=> sales distribution is far from a normal distribution (a positive long thin tail)
# Here, Sales that exceeds 1000 would be definitely considered as an outlier.

# Plot dataset: Profit Column
print(df['Profit'].describe())
plt.subplot(413)
plt.scatter(range(df.shape[0]), np.sort(df['Profit'].values))
plt.xlabel('index')
plt.ylabel('Profit')
plt.title("Profit distribution")
sns.despine()
plt.subplot(414)
sns.distplot(df['Profit'])
plt.title("Distribution of Profit")
sns.despine()

print("Skewness: %f" % df['Profit'].skew())
print("Kurtosis: %f" % df['Profit'].kurt())
# positive tail>negative tail => positive skewed

## Univariate Anomaly Detection ===============================================
## Isolation Forest on Sales
isolation_forest = IsolationForest(n_estimators=100) #n_estimators : no. of base estimators in the ensemble
isolation_forest.fit(df['Sales'].values.reshape(-1, 1))
xx = np.linspace(df['Sales'].min(), df['Sales'].max(), len(df)).reshape(-1,1)
anomaly_score = isolation_forest.decision_function(xx)
outlier = isolation_forest.predict(xx)
plt.figure(figsize=(10,4))
plt.plot(xx, anomaly_score, label='anomaly score')
plt.fill_between(xx.T[0], np.min(anomaly_score), np.max(anomaly_score), 
                 where=outlier==-1, color='r', 
                 alpha=.4, label='outlier region')
plt.legend()
plt.ylabel('anomaly score')
plt.xlabel('Sales')
plt.show();

# Visually investigate one anomaly
#print(df.iloc[10])

## Isolation Forest on Profit
isolation_forest = IsolationForest(n_estimators=100)
isolation_forest.fit(df['Profit'].values.reshape(-1, 1))
xx = np.linspace(df['Profit'].min(), df['Profit'].max(), len(df)).reshape(-1,1)
anomaly_score = isolation_forest.decision_function(xx)
outlier = isolation_forest.predict(xx)
plt.figure(figsize=(10,4))
plt.plot(xx, anomaly_score, label='anomaly score')
plt.fill_between(xx.T[0], np.min(anomaly_score), np.max(anomaly_score), 
                 where=outlier==-1, color='r', 
                 alpha=.4, label='outlier region')
plt.legend()
plt.ylabel('anomaly score')
plt.xlabel('Profit')
plt.show();

# Visually investigate one anomaly
#print(df.iloc[3])  # Any negative profit would be an anomaly
#print(df.iloc[1])
