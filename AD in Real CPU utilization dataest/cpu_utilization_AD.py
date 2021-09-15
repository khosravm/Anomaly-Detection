"""
Time Series Anomaly Detection(AD):
    General idea: Find out what is normal and if something deviates too much 
    from it — this is an anomaly.
   
Some Possible approaches:
    - Statistical Methods => ARIMA
    - Deviations from association rules and frequent itemsets
    - One-class Support Vector Machine
    - Clustering-based techniques (k-means)
    - Density-based techniques (k-nearest neighbor, local outlier factor)
    - Autoencoders and replicators (Neural Networks)

Datasets: Real CPU utilization from AWS Cloudwatch metrics for Amazon 
          Relational Database Service. Load from NAB (The Numenta Anomaly Benchmark)
 
Libraries:
        - Scikit-learn => data preprocessing
        - Statsmodel => ARIMA model
        - PyTorch => neural networks (here CNN and LSTM)
        - Plotly => plots and graphs
        
1?- What is normal?
    1. Given the value of CPU usage, try to reconstruct it. This task will be 
       given to the LSTM model. (reconstruct)
    2. Given the values of CPU usage, try to predict the next value. This task 
       will be given to the ARIMA and CNN models. (predict the next value)
    => If the model reconstructs or predicts easily (meaning, with little 
       deviation), then, it is normal data.

2?- How to measure deviation?
    Absolute error for ARIMA and Square error for 2 other methods
    
3?- how to pick the threshold?
    Use the 'three-sigma statistic rule' by measuring the mean and the standard 
    deviation of the errors in all training data (and only training, because  
    validation data only use to see how model performs). And then calculate the 
    threshold, above which deviation is “too much”.
    
Models:
1- ARIMA statistical model — predicts next value
    Implementation of this model is not so interesting since all we can play 
    with are hyperparameters for this model. To find the model that produces 
    the best predictions we will iterate over hyperparameters and will pick 
    the combination.
    Note: one important point about ARIMA is that time for training and 
    optimizing the coefficients takes ~10 times longer than it takes to 
    complete the same training of both neural networks.

2- Convolutional Neural Network — predicts next value
3- LSTM Neural Network — reconstructs current value
@author: Mahdieh khosravi
"""
# from pathlib import Path
from plotly.offline import plot
import plotly.graph_objects as go 
import numpy as np 
import pandas as pd 
import json # Since Anomalies' timestamps are in json format
import torch 
import torch.nn as nn
import torch.optim as opt 
from tqdm import tqdm # tqdm is a Python library for adding progress bar
from torch.utils.data import Dataset, DataLoader 

## Load Data ==================================================================

# Path from data folder to the training file
training_filename = 'realAWSCloudwatch/rds_cpu_utilization_cc0c53.csv'

# Path from data folder to the validation file
valid_filename = 'realAWSCloudwatch/rds_cpu_utilization_e47b3b.csv'

with open('combined_labels.json', 'r') as f:
    anomalies_timestamps = json.load(f)
    
train = pd.read_csv('rds_cpu_utilization_cc0c53.csv')
valid = pd.read_csv('rds_cpu_utilization_e47b3b.csv')
print(train.head())

## Data Preprocessing  ========================================================
from sklearn.preprocessing import StandardScaler

# Standardization: rescaling the numbers to mean = 0 and stnd. deviation = 1
# Make function for further usage
def parse_and_standardize(df: pd.DataFrame, scaler: StandardScaler = None):
    df['timestamp']   = pd.to_datetime(df['timestamp'])
    df['stand_value'] = df['value']
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(df['stand_value'].values.reshape(-1, 1))
    df['stand_value'] = scaler.transform(df['stand_value'].values.reshape(-1, 1))
    return scaler

data_scaler = parse_and_standardize(train)
parse_and_standardize(valid, data_scaler)

## extract anomalies into dedicated variables (training and validation dataset)
train_anomalies = train[train['timestamp'].isin(anomalies_timestamps[training_filename])]
valid_anomalies = valid[valid['timestamp'].isin(anomalies_timestamps[valid_filename])]
print('\n')
print(train_anomalies)
print(valid_anomalies)
"""
## Visualization ==============================================================
# Prepare layout w/ titles
layout = dict(xaxis=dict(title='Timestamp'), yaxis=dict(title='CPU Utilization')) 

# Create the figure for plotting the data
fig = go.Figure(layout=layout) 

# Add non-anomaly data to the figure
fig.add_trace(go.Scatter(x=train['timestamp'], y=train['value'], 
                         mode='markers', name='Non-anomaly',
                         marker=dict(color='blue')))

# Add anomaly data to the figure
fig.add_trace(go.Scatter(x=train_anomalies['timestamp'],
                         y=train_anomalies['value'], 
                         mode='markers', name='Anomaly',
                         marker=dict(color='green', size=13)))
plot(fig) 

ig = go.Figure()
fig.add_trace(go.Scatter(x=valid['timestamp'], y=valid['value'], 
                         mode='markers', name='Non-anomaly',
                         marker=dict(color='blue')))
fig.add_trace(go.Scatter(x=valid_anomalies['timestamp'],
                         y=valid_anomalies['value'], 
                         mode='markers', name='Anomaly',
                         marker=dict(color='green', size=13)))
plot(fig) 
"""
#********************************* AD Models **********************************
## 1. SARIMAX — A modified ARIMA model that adds seasonal causes into ARIMA===

import statsmodels.api as sm
from itertools import product

def write_predict(train_df: pd.DataFrame, valid_df: pd.DataFrame):
    # Initial approximation of parameters
    Qs = range(0, 2)
    qs = range(0, 3)
    Ps = range(0, 3)
    ps = range(0, 3)
    D=1
    d=1
    parameters = product(ps, qs, Ps, Qs)
    parameters_list = list(parameters)
    
    # Best Model Selection
    results = []
    best_aic = float("inf")
    for param in parameters_list:
        try:
            model=sm.tsa.statespace.SARIMAX(
                train_df.value, order=(param[0], d, param[1]),
                seasonal_order=(param[2], D, param[3], 12),
                initialization='approximate_diffuse'
                ).fit()
        except ValueError:
            print('wrong parameters:', param)
            continue
        aic = model.aic
        if aic < best_aic:
            best_model = model
            best_aic = aic
            best_param = param
        results.append([param, model.aic])
    
    # Writing of the predictions for training data
    train_df['predict'] = best_model.predict()
    train_df['predict'].fillna(0, inplace=True)
    
    # Writing of the predictions for validation data
    best_model_valid = sm.tsa.statespace.SARIMAX(
        valid_df.value, order=(best_param[0], d, best_param[1]),
        seasonal_order=(best_param[2], D, best_param[3], 12),
        initialization='approximate_diffuse'
        ).fit()
    valid_df['predict'] = best_model_valid.predict()
    valid_df['predict'].fillna(0, inplace=True)
    
    

# Calling of the function
write_predict(train, valid)

# Create the figure for plotting the data
fig = go.Figure(layout=layout) 

# Add real data to the figure
fig.add_trace(go.Scatter(x=train['timestamp'], y=train['value'], 
                         mode='markers', name='Ground Truth',
                         marker=dict(color='blue')))

# Add predicted data to the figure
fig.add_trace(go.Scatter(x=train['timestamp'],
                         y=train['predict'], 
                         mode='markers', name='Predicted Values',
                         marker=dict(color='green', size=13)))
plot(fig)

ig = go.Figure()
fig.add_trace(go.Scatter(x=valid['timestamp'], y=valid['value'], 
                         mode='markers', name='Ground Truth',
                         marker=dict(color='blue')))
fig.add_trace(go.Scatter(x=valid['timestamp'],
                         y=valid['predict'], 
                         mode='markers', name='Predicted Values',
                         marker=dict(color='green', size=13)))
plot(fig) 

## 2. CNN =====================================================================


# Dataset - the base class to be inherited
from torch.utils.data import Dataset, DataLoader 
# DataLoader: it used later for the training process

# wrap data with custom class that just inherits from PyTorch’s Dataset
class CPUDataset(Dataset):
    def __init__(self, data: pd.DataFrame, size: int, 
                 step: int = 1):
        self.chunks = torch.FloatTensor(data['stand_value']).unfold(0, size+1, step)
        self.chunks = self.chunks.view(-1, 1, size+1)
    def __len__(self):
        return self.chunks.size(0)
    
    def __getitem__(self, i):
        x = self.chunks[i, :, :-1]
        y = self.chunks[i, :, -1:].squeeze(1)
        return x, y
    
n_factors = 10 # The No. of previous values that are to be used for the prediction of the next one
train_ds = CPUDataset(train, n_factors)
valid_ds = CPUDataset(valid, n_factors)

# Convolution layer
# here PyTorch has all neural net functions and activations

def conv_layer(in_feat, out_feat, kernel_size=3, stride=1,
               padding=1, relu=True):
    res = [
        nn.Conv1d(in_feat, out_feat, kernel_size=kernel_size,
                  stride=stride, padding=padding, bias=False),
        nn.BatchNorm1d(out_feat),
    ]
    if relu:
        res.append(nn.ReLU())
    return nn.Sequential(*res)

# Batch Normalization
class ResBlock(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()
        self.in_feat, self.out_feat = in_feat, out_feat
        self.conv1 = conv_layer(in_feat, out_feat)
        self.conv2 = conv_layer(out_feat, out_feat, relu=False)
        if self.apply_shortcut:
            self.shortcut = conv_layer(in_feat, out_feat,
                                       kernel_size=1, padding=0,
                                       relu=False)
    
    def forward(self, x):
        out = self.conv1(x)
        if self.apply_shortcut:
            x = self.shortcut(x)
        return x + self.conv2(out)
    
    @property
    def apply_shortcut(self):
        return self.in_feat != self.out_feat
    
# concatenate Average Pooling and Max Pooling
class AdaptiveConcatPool1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool1d(1)
        self.mp = nn.AdaptiveMaxPool1d(1)
    
    def forward(self, x): 
        return torch.cat([self.mp(x), self.ap(x)], 1)
    
class CNN(nn.Module):
    def __init__(self, out_size):
        super().__init__()
        self.base = nn.Sequential(
            ResBlock(1, 8), #shape = batch, 8, n_factors
            ResBlock(8, 8), 
            ResBlock(8, 16), #shape = batch, 16, n_factors
            ResBlock(16, 16),
            ResBlock(16, 32), #shape = batch, 32, n_factors
            ResBlock(32, 32),
            ResBlock(32, 64), #shape = batch, 64, n_factors
            ResBlock(64, 64),
        )
        self.head = nn.Sequential(
            AdaptiveConcatPool1d(), #shape = batch, 128, 1
            nn.Flatten(),
            nn.Linear(128, out_size)
        )
        
    def forward(self, x):
        out = self.base(x)
        out = self.head(out)
        return out
   
def train_model(model: CNN, dataloaders: dict, optimizer: opt.Optimizer, 
                scheduler, criterion, device: torch.device, epochs: int):
    losses_data = {'train': [], 'valid': []}
    model.to(device)
    
    # Loop over epochs
    for epoch in tqdm(range(epochs)):
        print(f'Epoch {epoch}/{epochs-1}')
        
        # Training and validation phases
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.
            running_total = 0.
            
            # Loop over batches of data
            for idx, batch in tqdm(enumerate(dataloaders[phase]), 
                                   total=len(dataloaders[phase]), 
                                   leave=False
                                   ):
                x, y = batch
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    out = model(x)
                    loss = criterion(out, y)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                running_loss += loss.item() * y.size(0)
                running_total += y.size(0)

            epoch_loss = running_loss / running_total
            print(f'{phase.capitalize()} Loss: {epoch_loss}')
            losses_data[phase].append(epoch_loss)
    return losses_data

epochs = 50
cnn_model = CNN(out_size=1)
dataloaders = {
    'train': DataLoader(train_ds, batch_size=128, shuffle=True),
    'valid': DataLoader(valid_ds, batch_size=128)
}
optim = opt.Adam(cnn_model.parameters(), lr=1e-1, weight_decay=1e-3)
sched = opt.lr_scheduler.OneCycleLR(optim, max_lr=1e-3, steps_per_epoch=len(dataloaders['train']), epochs=epochs)
criterion = nn.MSELoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
losses = train_model(cnn_model, dataloaders, optim, sched, criterion, device, epochs)

# CNN Model Loss visualization
layout = dict(xaxis=dict(title='Epoch'), yaxis=dict(title='Loss'))
fig = go.Figure(layout=layout)
fig.add_trace(go.Scatter(y=losses['train'], mode='lines', name='Train Loss',))
fig.add_trace(go.Scatter(y=losses['valid'], mode='lines', name='Valid Loss'))
plot(fig)
# Switching model into evaluation mode
cnn_model = cnn_model.eval()

# Calculation of the predictions for training data
with torch.no_grad():
    res_train = cnn_model(train_ds[:][0].to(device))
res_train = res_train.cpu()

# Calculation of the predictions for validation data
with torch.no_grad():
    res_valid = cnn_model(valid_ds[:][0].to(device))
res_valid = res_valid.cpu()

# Prepare layout w/ titles
layout = dict(xaxis=dict(title='Timestamp'), yaxis=dict(title='CPU Utilization')) 
# Create the figure for plotting the data
fig = go.Figure(layout=layout) 

# Add real data to the figure
fig.add_trace(go.Scatter(x=train['timestamp'], y=train['stand_value'], 
                         mode='markers', name='Ground Truth',
                         marker=dict(color='blue')))

# Add predicted data to the figure
yt = pd.DataFrame(res_train.numpy())
fig.add_trace(go.Scatter(x = train['timestamp'],
                         y = yt[0], 
                         mode='markers', name='Reconstructed Values',
                         marker=dict(color='green', size=13)))
plot(fig)

ig = go.Figure()
fig.add_trace(go.Scatter(x=valid['timestamp'], y=valid['stand_value'], 
                         mode='markers', name='Ground Truth',
                         marker=dict(color='blue')))
yr = pd.DataFrame(res_valid.numpy())
fig.add_trace(go.Scatter(x=valid['timestamp'],
                         y = yr[0], 
                         mode='markers', name='Reconstructed Values',
                         marker=dict(color='green', size=13)))
plot(fig) 

## 3. LSTM ====================================================================

class CPUDataset(Dataset):
    def __init__(self, data: pd.DataFrame, size: int):
        self.chunks = torch.FloatTensor(data['stand_value']).unfold(0, size, size)
        
    def __len__(self):
        return self.chunks.size(0)
    
    def __getitem__(self, i):
        x = self.chunks[i]
        return x

train_ds = CPUDataset(train, 64)
valid_ds = CPUDataset(valid, 64)

# Define LSTM NN
class LSTMModel(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, device):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm        = nn.LSTM(in_size, hidden_size)
        self.linear      = nn.Linear(hidden_size, out_size)
        self.device      = device
        self.init_hidden()
        
    def forward(self, x):
        out, self.hidden_state = self.lstm(
            x.view(len(x), 1, -1), self.hidden_state
        )
        self.hidden_state = tuple(
            [h.detach() for h in self.hidden_state]
        )
        out = out.view(len(x), -1)
        out = self.linear(out)
        return out
    
    def init_hidden(self):
        self.hidden_state = (
            torch.zeros((1, 1, self.hidden_size)).to(self.device),
            torch.zeros((1, 1, self.hidden_size)).to(self.device))
        
# Train
def train_model(model: LSTMModel, dataloaders: dict, optimizer: opt.Optimizer, 
                scheduler, criterion, device: torch.device, epochs: int):
    losses_data = {'train': [], 'valid': []}
    model.to(device)
    for epoch in tqdm(range(epochs)):
        print(f'Epoch {epoch}/{epochs-1}')
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.
            running_total = 0.
            
        # Here changes start
            for idx, sequence in enumerate(dataloaders[phase]):
                value = sequence
                value = value.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    out  = model(value.view(-1, 1))
                    loss = criterion(out.view(-1), value.view(-1))
        # Here changes end

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                running_loss  += loss.item() * out.size(0)
                running_total += out.size(0)

            epoch_loss = running_loss / running_total
            print(f'{phase.capitalize()} Loss: {epoch_loss}')
            losses_data[phase].append(epoch_loss)
    return losses_data


epochs = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = LSTMModel(1, 128, 1, device)
dataloaders = {
    'train': DataLoader(train_ds, batch_size=1),
    'valid': DataLoader(valid_ds, batch_size=1)
}
optim = opt.Adam(params=model.parameters(), lr=1e-3)
sched = opt.lr_scheduler.OneCycleLR(
  optim, max_lr=1e-3, steps_per_epoch=len(dataloaders['train']), epochs=epochs)
criterion = nn.MSELoss()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

losses = train_model(model, dataloaders, optim, sched, criterion, device, epochs)

# Switching model into evaluation mode
model.eval()

with torch.no_grad():
    res_train = model(train_ds[:].flatten().to(device))
res_train = res_train.cpu()

# Calculation of the predictions for validation data
with torch.no_grad():
    res_valid = model(valid_ds[:].flatten().to(device))
res_valid = res_valid.cpu()

# Visualization ==============================================================
# Prepare layout w/ titles
layout = dict(xaxis=dict(title='Timestamp'), yaxis=dict(title='CPU Utilization')) 
# Create the figure for plotting the data
fig = go.Figure(layout=layout) 

# Add real data to the figure
fig.add_trace(go.Scatter(x=train['timestamp'], y=train['stand_value'], 
                         mode='markers', name='Ground Truth',
                         marker=dict(color='blue')))

# Add predicted data to the figure
yt = pd.DataFrame(res_train.numpy())
fig.add_trace(go.Scatter(x = train['timestamp'],
                         y = yt[0], 
                         mode='markers', name='Reconstructed Values',
                         marker=dict(color='green', size=13)))
plot(fig)

ig = go.Figure()
fig.add_trace(go.Scatter(x=valid['timestamp'], y=valid['stand_value'], 
                         mode='markers', name='Ground Truth',
                         marker=dict(color='blue')))
yr = pd.DataFrame(res_valid.numpy())
fig.add_trace(go.Scatter(x=valid['timestamp'],
                         y = yr[0], 
                         mode='markers', name='Reconstructed Values',
                         marker=dict(color='green', size=13)))
plot(fig) 

