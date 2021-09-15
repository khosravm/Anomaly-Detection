"""
# Application of anomaly detection in the Manufacturing industry (Sensor data)

- In time series analysis, it is important that the data is stationary (The 
mean and standard deviation of the data do not change over time) and have 
no autocorrelation.
So => (1) perform the Dickey Fuller test to quantitatively verify the observed 
          stationarity
      (2) inspect the autocorrelation of the features  
before feeding them into the clustering algorithms to detect anomalies.

- It is pretty computationally expensive to train models with all of the 52 
sensors/features and it is not efficient. Therefore, we employ Principal 
Component Analysis (PCA) technique to extract new features to be used for the 
modeling. In order to properly apply PCA, the data must be scaled and 
standardized. This is because PCA and most of the learning algorithms are 
distance based algorithms.

- 3 methods used for Modeling:
    [1] Benchmark model: Interquartile Range (IQR)
        - Calculate IQR which is the difference between 75th (Q3)and 25th (Q1) 
          percentiles.
        - Calculate upper and lower bounds for the outlier.
        - Filter the data points that fall outside the upper and lower bounds 
        and flag them as outliers.
        - Finally, plot the outliers on top of the time series data (the 
          readings from sensor_11 in this case)
    
    [2] K-Means clustering
        - Calculate the distance between each point and its nearest centroid. 
        The biggest distances are considered as anomaly.
        - We use outliers_fraction to provide information to the algorithm 
        about the proportion of the outliers present in our data set. Situations may vary from data set to data set. However, as a starting figure, I estimate outliers_fraction=0.13 (13% of df are outliers as depicted).
        - Calculate number_of_outliers using outliers_fraction.
        - Set threshold as the minimum distance of these outliers.
        - The anomaly result of anomaly1 contains the above method Cluster 
        (0:normal, 1:anomaly).
        - Visualize anomalies with Time Series view.
    
    [3] Isolation Forest


@author: Mahdieh khosravi
"""
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('Sensor.csv')
# print(df.info())

### Data Prepearation *********************************************************

## Drop duplicates
df_tidy = df.drop_duplicates()
# Entire "sensor_15" column is NaN therefore remove it from data
del df_tidy['sensor_15']
# To ignore deprecation warnings
import warnings
warnings.filterwarnings("ignore")
# Convert the data type of timestamp column to datatime format
df_tidy['date'] = pd.to_datetime(df_tidy['timestamp'])
del df_tidy['timestamp']

## Handling Missing values
# Function that calculates the percentage of missing values
def calc_percent_NAs(df):
    nans = pd.DataFrame(df.isnull().sum().sort_values(ascending=False)/len(df), columns=['percent']) 
    idx  = nans['percent'] > 0
    return nans[idx]
# Let's use above function to look at top ten columns with NaNs
print(calc_percent_NAs(df_tidy).head(10))

df['sensor_15'] = 0
aa = calc_percent_NAs(df_tidy)
b  = aa.index
df_tidy = df_tidy.dropna(subset=b[7:]) # drop missing valus with percentage less than 0.001675
df_tidy = df_tidy.fillna(df_tidy.mean()) # replace the rest with mean()

## Exploratory Data Analysis (EDA)
# Extract the readings from the BROKEN state of the pump
broken = df_tidy[df_tidy['machine_status']=='BROKEN']
# Extract the names of the numerical columns
df2   = df_tidy.drop(['machine_status'], axis=1)
names = df2.columns
# Plot time series for each sensor with BROKEN state marked with X in red color
for name in names:
    _ = plt.figure(figsize=(18,3))
    _ = plt.plot(broken[name], linestyle='none', marker='X', color='red', markersize=12)
    _ = plt.plot(df2[name], color='blue')
    _ = plt.title(name)
    plt.show()
    
## Stationarity and Autocorrelation
# Resample the entire dataset by daily average (rule='D')
rollmean = df_tidy.resample('D', on='date').mean()
rollstd  = df_tidy.resample('D', on='date').std()
# Plot time series for each sensor with its mean and standard deviation
for name in names[:-1]:
    _ = plt.figure(figsize=(18,3))
    _ = plt.plot(df_tidy[name], color='blue', label='Original')
    _ = plt.plot(rollmean[name], color='red', label='Rolling Mean')
    _ = plt.plot(rollstd[name], color='black', label='Rolling Std' )
    _ = plt.legend(loc='best')
    _ = plt.title(name)
    plt.show()
    
### Pre-Processing and Dimensionality Reduction *******************************

# Standardize/scale the dataset and apply PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
# Extract the names of the numerical columns
# df2    = df_tidy.drop(['machine_status'], axis=1)
df2    = df2.dropna()
names  = df2.columns
x      = df2[names[:-1]]
scaler = StandardScaler()
pca    = PCA()
pipeline = make_pipeline(scaler, pca)
pipeline.fit(x)
# Plot the principal components against their inertia
features = range(pca.n_components_)
_ = plt.figure(figsize=(15, 5))
_ = plt.bar(features, pca.explained_variance_)
_ = plt.xlabel('PCA feature')
_ = plt.ylabel('Variance')
_ = plt.xticks(features)
_ = plt.title("Importance of the Principal Components based on inertia")
plt.show()

# Since the first two principal components are the most important as per the 
# features extracted by the PCA in above importance plot =>>
# Calculate PCA with 2 components
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2'])
"""
# Check again the stationarity and autocor. of these 2 principal components
from statsmodels.tsa.stattools import adfuller
# Run Augmented Dickey Fuller Test (To see it is stationary)
result = adfuller(principalDf['pc1'])
# Print p-value
print(result[1])

# Plot ACF (To check that there is no autocorrelation)
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(principalDf['pc1'].dropna(), lags=20, alpha=0.05)
"""

### Modeling ******************************************************************
### [1] Interquartile Range

# Calculate IQR for the 1st principal component (pc1)
df3 = principalDf
df0 = df_tidy
q1_pc1, q3_pc1 = df3['pc1'].quantile([0.25, 0.75])
iqr_pc1 = q3_pc1 - q1_pc1
# Calculate upper and lower bounds for outlier for pc1
lower_pc1 = q1_pc1 - (1.5*iqr_pc1)
upper_pc1 = q3_pc1 + (1.5*iqr_pc1)
# Filter out the outliers from the pc1

df0['anomaly_pc1'] = ((df3['pc1']>upper_pc1) | (df3['pc1']<lower_pc1)).astype('int')
# Calculate IQR for the 2nd principal component (pc2)
q1_pc2, q3_pc2 = df3['pc2'].quantile([0.25, 0.75])
iqr_pc2 = q3_pc2 - q1_pc2
# Calculate upper and lower bounds for outlier for pc2
lower_pc2 = q1_pc2 - (1.5*iqr_pc2)
upper_pc2 = q3_pc2 + (1.5*iqr_pc2)
# Filter out the outliers from the pc2
df0['anomaly_pc2'] = ((df3['pc2']>upper_pc2) | (df3['pc2']<lower_pc2)).astype('int')
# Let's plot the outliers from pc1 on top of the sensor_11 and see where they occured in the time series
a = df0[df0['anomaly_pc2'] == 1] #anomaly
_ = plt.figure(figsize=(18,6))
_ = plt.plot(df0['sensor_11'], color='blue', label='Normal')
_ = plt.plot(a['sensor_11'], linestyle='none', marker='X', color='red', markersize=12, label='Anomaly')
_ = plt.xlabel('Date and Time')
_ = plt.ylabel('Sensor Reading')
_ = plt.title('Sensor_11 Anomalies')
_ = plt.legend(loc='best')
plt.show();

### [2] K-Means Clustering
# Import necessary libraries
import numpy as np
from sklearn.cluster import KMeans
# Start k-means clustering with k=2 as already know that there are 3 classes of "NORMAL" vs 
# "NOT NORMAL" which are combination of BROKEN" and "RECOVERING"
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(principalDf.values)
labels = kmeans.predict(principalDf.values)
unique_elements, counts_elements = np.unique(labels, return_counts=True)
clusters = np.asarray((unique_elements, counts_elements))
# Write a function that calculates distance between each point and the centroid of the closest cluster
def getDistanceByPoint(data, model):
    """ Function that calculates the distance between a point and centroid of a cluster, 
            returns the distances in pandas series"""
    distance = []
    for i in range(0,len(data)):
        Xa = np.array(data.loc[i])
        Xb = model.cluster_centers_[model.labels_[i]-1]
        distance.append(np.linalg.norm(Xa-Xb))
    return pd.Series(distance, index=data.index)
# Assume that 13% of the entire data set are anomalies 
outliers_fraction = 0.13
# get the distance between each point and its nearest centroid. The biggest distances are considered as anomaly
distance = getDistanceByPoint(principalDf, kmeans)
# number of observations that equate to the 13% of the entire data set
number_of_outliers = int(outliers_fraction*len(distance))
# Take the minimum of the largest 13% of the distances as the threshold
threshold = distance.nlargest(number_of_outliers).min()
# anomaly1 contain the anomaly result of the above method Cluster (0:normal, 1:anomaly) 
principalDf['anomaly1'] = (distance >= threshold).astype(int)
print(principalDf['anomaly1'].value_counts())

### [3] Isolation Forest
# Import IsolationForest
from sklearn.ensemble import IsolationForest
# Assume that 13% of the entire data set are anomalies
 
outliers_fraction = 0.13
model =  IsolationForest(contamination=outliers_fraction)
model.fit(principalDf.values) 
principalDf['anomaly2'] = pd.Series(model.predict(principalDf.values))
print(principalDf['anomaly2'].value_counts())
# visualization
df0 = df_tidy
df0['anomaly2'] = pd.Series(principalDf['anomaly2'].values, index=df0.index)
a = df0.loc[df0['anomaly2'] == -1] #anomaly
_ = plt.figure(figsize=(18,6))
_ = plt.plot(df0['sensor_11'], color='blue', label='Normal')
_ = plt.plot(a['sensor_11'], linestyle='none', marker='X', color='red', markersize=12, label='Anomaly')
_ = plt.xlabel('Date and Time')
_ = plt.ylabel('Sensor Reading')
_ = plt.title('Sensor_11 Anomalies')
_ = plt.legend(loc='best')
plt.show();
