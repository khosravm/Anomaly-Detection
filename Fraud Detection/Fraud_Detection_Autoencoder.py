"""
Fraud Detection Using Autoencoder:

Fraud detection is a process that detects & prevents fraudsters from obtaining 
money or property through false means. It is a set of activities undertaken to 
detect and block the attempt of fraudsters from obtaining money or property 
fraudulently. Fraud detection is prevalent across banking, insurance, medical, 
government, and public sectors, as well as in law enforcement agencies.     

Autoencoders are artificial neural networks capable of learning efficient 
representations of the input data. This learned representation is called 
‘coding’ and this learning happens unsupervised.

Autoencoder consisting of 4 fully connected layers with 14,7,7,29 neurons: 
First 2 layers act as encoder and last 2 layers act as decoder. 
Note- last layer has 29 nodes corresponding to 29 feature in the input data 
item. 
    
In order to predict whether or not a new/unseen transaction is normal or 
fraudulent, we'll calculate the reconstruction error from the transaction data 
itself. If the error is larger than a predefined threshold, we'll mark it as a 
fraud (since our model should have a low error on normal transactions). 

ROC - For the ROC curve, the ideal curve is close to the top left: to have a 
model that produces a high recall while keeping a low false positive rate.

Setting a threshold that is used to make a classification decision in a model 
is a way to adjust the trade-off of precision and recall for a given classifier. 
From the curve obtained for the case under consideration (here the optimal 
value is around 0.29.)

Precision vs. Recall - A high area under the curve represents both high recall 
and high precision, where high precision relates to a low false positive rate, 
and high recall relates to a low false negative rate. High scores for both show 
that the classifier is returning accurate results (high precision), as well as 
returning a majority of all positive results (high recall).
@author: Mahdieh khosrav
"""
# Requirements ----------------------------------------------------------------
import pandas as pd
import numpy as np
# import pickle

# from torch.autograd import Variable
import torch
import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader


import torch.utils.data as data_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams

from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)

sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 14, 8

RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]

# Data Loading and Preprocessing ----------------------------------------------
df = pd.read_csv('creditcard.csv')
# Data exploration
print(df.head(5))
print(df.shape)
print(df.describe())
print('\n\n')
print('Is there any missing value? ', df.isnull().values.any())

# Class distribution ( 0 - non fraudulent, 1 - fraudulent)
print('\n\n')
print('Value counts for class features: ',df['Class'].value_counts())
# Visualization
count_classes = pd.value_counts(df['Class'], sort = True)
count_classes.plot(kind = 'bar', rot=0, color="g")
plt.title("Normal vs Fraudulant Transactions")
plt.xticks(range(2), LABELS)
plt.xlabel("Transaction Class")
plt.ylabel("Frequency")

# More expolaration to deal with imbalanced (in class column) dataset problem
fraudsDF = df[df.Class == 1]
normalDF = df[df.Class == 0]

print(fraudsDF.shape)
print(fraudsDF.shape)
print(fraudsDF.Amount.describe())
print(normalDF.Amount.describe())

# Graphical Exploration
# Histogram
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Amount per transaction by class')

bins = 50

ax1.hist(fraudsDF.Amount, bins = bins)
ax1.set_title('Fraud')

ax2.hist(normalDF.Amount, bins = bins)
ax2.set_title('Normal')

plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.xlim((0, 20000))
plt.yscale('log')
plt.show()

# Scattering
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Time of transaction vs Amount by class')

ax1.scatter(fraudsDF.Time, fraudsDF.Amount)
ax1.set_title('Fraud')

ax2.scatter(normalDF.Time, normalDF.Amount)
ax2.set_title('Normal')

plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
plt.show()

## Preprocessing

# Because it has no bearing on fraudulent activity:
transactionData = df.drop(['Time'], axis=1) 
#Standardization
transactionData['Amount'] = StandardScaler().fit_transform(transactionData['Amount'].values.reshape(-1, 1))

# Split dataset
X_train, X_test = train_test_split(transactionData, test_size=0.2, random_state=RANDOM_SEED)

# Keep non-fradual and then remove class features
X_train = X_train[X_train.Class == 0]
X_train = X_train.drop(['Class'], axis=1)
#print(type(X_train))

y_test = X_test['Class']
X_test = X_test.drop(['Class'], axis=1)

X_train = X_train.values # df to array
#print(type(X_train))
X_test = X_test.values
y_test = y_test.values
# print(y_test.size)

# Model -----------------------------------------------------------------------
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(29, 14),
            nn.Tanh(),
            nn.Linear(14, 7),
            nn.LeakyReLU(),
            )
        
        self.decoder = nn.Sequential(
           nn.Linear(7, 7),
           nn.Tanh(),
           nn.Linear(7, 29),
           nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
model = Autoencoder().double().cpu()

### Train Model ---------------------------------------------------------------
num_epochs = 100
minibatch_size = 32
learning_rate = 1e-3

train_loader = data_utils.DataLoader(X_train, batch_size=minibatch_size, shuffle=True)
#test = data_utils.TensorDataset(torch.from_numpy(X_test).double(),torch.from_numpy(y_test).double())
#test_loader = data_utils.DataLoader(test, batch_size=minibatch_size, shuffle=True)
test_loader = data_utils.DataLoader(X_test, batch_size=1, shuffle=False)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=10e-05)

history = {}
history['train_loss'] = []
history['test_loss'] = []

for epoch in range(num_epochs):
    h = np.array([])
    for data in train_loader:
        #print(type(data))
        #data = Variable(data).cpu()
        #print(type(data))
        # ===================forward=====================
        output = model(data)
        loss   = criterion(output, data)
        h      = np.append(h, loss.item())
        
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    mean_loss = np.mean(h)
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, num_epochs, mean_loss))
    history['train_loss'].append(mean_loss)
    

torch.save(model.state_dict(), './credit_card_model.pth')

# Plot training Results
#history['train_loss']
#plt.plot(range(num_epochs),history['train_loss'],'ro',linewidth=2.0)
plt.plot(history['train_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.axis([0,100,0.69,0.80])
#plt.legend(['train', 'test'], loc='upper right');
plt.show()

### Evaluate Model ------------------------------------------------------------

pred_losses = {'pred_loss' : []}
model.eval()
with torch.no_grad():
   # test_loss = 0
    for data in test_loader:
        inputs = data
        # print(inputs)
        outputs = model(inputs)
        loss = criterion(outputs, inputs).data.item()
        #print(loss)
        pred_losses['pred_loss'].append(loss)
        #pred_losses = model([y_test.size, y_test])
reconstructionErrorDF = pd.DataFrame(pred_losses)
reconstructionErrorDF['Class'] = y_test

print('\n\n')
print('Reconstruction Error:', reconstructionErrorDF.describe())

## Plot Evaluation Results
## Plot Reconstruction Errors 
# Without Fraud
fig = plt.figure()
ax = fig.add_subplot(111)
normal_error_df = reconstructionErrorDF[(reconstructionErrorDF['Class']== 0) & (reconstructionErrorDF['pred_loss'] < 10)]
_ = ax.hist(normal_error_df.pred_loss.values, bins=10)
fig.suptitle('Reconstruction Errors without Fraud', fontsize=16)
          
# With Fraud
fig = plt.figure()
ax = fig.add_subplot(111)
fraud_error_df = reconstructionErrorDF[(reconstructionErrorDF['Class']== 1) ]
_ = ax.hist(fraud_error_df.pred_loss.values, bins=10)
fig.suptitle('Reconstruction Errors with Fraud', fontsize=16)

# ROC Curves
fpr, tpr, thresholds = roc_curve(reconstructionErrorDF.Class, reconstructionErrorDF.pred_loss)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.001, 1])
plt.ylim([0, 1.001])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Precision vs Recall
precision, recall, th = precision_recall_curve(reconstructionErrorDF.Class, reconstructionErrorDF.pred_loss)
plt.plot(recall, precision, 'b', label='Precision-Recall curve')
plt.title('Recall vs Precision')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()
# Precision for different threshold values
plt.plot(th, precision[1:], 'b', label='Threshold-Precision curve')
plt.title('Precision for different threshold values')
plt.xlabel('Threshold')
plt.ylabel('Precision')
plt.show()
# Recall for different threshold values
plt.plot(th, recall[1:], 'b', label='Threshold-Recall curve')
plt.title('Recall for different threshold values')
plt.xlabel('Threshold')
plt.ylabel('Recall')
plt.show()

### Makeing prediction --------------------------------------------------------
threshold = 2.9

# Plot Reconstruction error for different classes
groups = reconstructionErrorDF.groupby('Class')
fig, ax = plt.subplots()

for name, group in groups:
    ax.plot(group.index, group.pred_loss, marker='o', ms=3.5, linestyle='',
            label= "Fraud" if name == 1 else "Normal")
ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
plt.show()

# Confusion matrix
y_pred = [1 if e > threshold else 0 for e in reconstructionErrorDF.pred_loss.values]
conf_matrix = confusion_matrix(reconstructionErrorDF.Class, y_pred)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d", 
            cmap=plt.cm.get_cmap('Blues'));
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

# To have better results (Reduce No. of normal transactions classified as 
# frauds) =>> adjust the threshold 


