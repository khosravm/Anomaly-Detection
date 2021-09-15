"""
Detect anomalies in Time Series data using an LSTM Autoencoder:
    Dataset: ECG (electrocardiogram) is a test that checks how the heart is 
    functioning by measuring the electrical activity of the heart.
    5 types of hearbeats (classes):
        - Normal (N)
        - R-on-T Premature Ventricular Contraction (R-on-T PVC)
        - Premature Ventricular Contraction (PVC)
        - Supra-ventricular Premature or Ectopic Beat (SP or EB)
        - Unclassified Beat (UB).
        =>> class_names = ['Normal','R on T','PVC','SP','UB']

Autoencoders try to learn only the most important features (compressed version) 
of the data. When training an Autoencoder, the objective is to reconstruct the 
input as best as possible. This is done by minimizing a loss function (just 
like in supervised learning). This function is known as reconstruction loss. 
Cross-entropy loss and Mean squared error are common examples. We’ll use normal 
heartbeats as training data for our model and record the reconstruction loss.

To classify a sequence as normal or an anomaly, we’ll pick a threshold above 
which a heartbeat is considered abnormal.
@author: Mahdieh khosravi
"""
# Prepare a dataset for Anomaly Detection from Time Series Data ---------------
import torch
from arff2pandas import a2p
# import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open('ECG5000_TRAIN.arff') as f:
  train = a2p.load(f)
with open('ECG5000_TEST.arff') as f:
  test = a2p.load(f)

# Combine the train & test data into a single data frame=> more data to train in Autoencoder
df = train.append(test)
df = df.sample(frac=1.0)
print(df.shape)

# Name the possible classes
CLASS_NORMAL = 1
class_names = ['Normal','R on T','PVC','SP','UB']

# Rename the last column to target
new_columns     = list(df.columns)
new_columns[-1] = 'target'
df.columns      = new_columns

# Exploratory Data Analysis (EDA)

# How many examples for each heartbeat class do we have?
print(df.target.value_counts())

# Visualizations
import matplotlib.pyplot as plt

x = list(df.target.unique().astype(int))
y = list(df.target.value_counts())

_ = plt.figure(figsize=(15, 5))
_ = plt.bar(x, y)
_ = plt.xlabel('PNG')
_ = plt.ylabel('Count')
_ = plt.xticks(x,class_names)
_ = plt.title("The normal class, has by far, the most examples!")
plt.show()

# Data Prepparation
# Get all normal heartbeats and drop the target (class) column:
normal_df = df[df.target == str(CLASS_NORMAL)].drop(labels='target', axis=1)
# Merge all other classes and mark them as anomalies:
anomaly_df = df[df.target != str(CLASS_NORMAL)].drop(labels='target', axis=1)

from sklearn.model_selection import train_test_split
# sklearn.utils.check_random_state()
import numpy as np
RANDOM_SEED = 1 #np.random.seed(1234)
train_df, val_df = train_test_split(
  normal_df,
  test_size=0.15,
  random_state= RANDOM_SEED
)
val_df, test_df = train_test_split(
  val_df,
  test_size=0.33,
  random_state= RANDOM_SEED
)

# Convert data into tensors
def create_dataset(df):
  sequences = df.astype(np.float32).to_numpy().tolist()
  dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
  n_seq, seq_len, n_features = torch.stack(dataset).shape
  return dataset, seq_len, n_features

# TS will be converted to a 2D Tensor in the shape sequence length x number of 
# features (140x1 in this case)
train_dataset, seq_len, n_features = create_dataset(train_df)
val_dataset, _, _ = create_dataset(val_df)
test_normal_dataset, _, _ = create_dataset(test_df)
test_anomaly_dataset, _, _ = create_dataset(anomaly_df)

# Build an LSTM Autoencoder with PyTorch --------------------------------------
import torch.nn as nn
class Encoder(nn.Module):
  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(Encoder, self).__init__()
    self.seq_len, self.n_features = seq_len, n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
    self.rnn1 = nn.LSTM(
      input_size=n_features,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )
    self.rnn2 = nn.LSTM(
      input_size=self.hidden_dim,
      hidden_size=embedding_dim,
      num_layers=1,
      batch_first=True
    )
  def forward(self, x):
    x = x.reshape((1, self.seq_len, self.n_features))
    x, (_, _) = self.rnn1(x)
    x, (hidden_n, _) = self.rnn2(x)
    return hidden_n.reshape((self.n_features, self.embedding_dim))

class Decoder(nn.Module):
  def __init__(self, seq_len, input_dim=64, n_features=1):
    super(Decoder, self).__init__()
    self.seq_len, self.input_dim = seq_len, input_dim
    self.hidden_dim, self.n_features = 2 * input_dim, n_features
    self.rnn1 = nn.LSTM(
      input_size=input_dim,
      hidden_size=input_dim,
      num_layers=1,
      batch_first=True
    )
    self.rnn2 = nn.LSTM(
      input_size=input_dim,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )
    self.output_layer = nn.Linear(self.hidden_dim, n_features)
  def forward(self, x):
    x = x.repeat(self.seq_len, self.n_features)
    x = x.reshape((self.n_features, self.seq_len, self.input_dim))
    x, (hidden_n, cell_n) = self.rnn1(x)
    x, (hidden_n, cell_n) = self.rnn2(x)
    x = x.reshape((self.seq_len, self.hidden_dim))
    return self.output_layer(x)

class RecurrentAutoencoder(nn.Module):
  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(RecurrentAutoencoder, self).__init__()
    self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
    self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)
  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x

model = RecurrentAutoencoder(seq_len, n_features, 128)
model = model.to(device)

# Train and evaluate your model -----------------------------------------------
import copy
def train_model(model, train_dataset, val_dataset, n_epochs):
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  criterion = nn.L1Loss(reduction='sum').to(device)
  history = dict(train=[], val=[])
  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 10000.0
  for epoch in range(1, n_epochs + 1):
    model = model.train()
    train_losses = []
    for seq_true in train_dataset:
      optimizer.zero_grad()
      seq_true = seq_true.to(device)
      seq_pred = model(seq_true)
      loss = criterion(seq_pred, seq_true)
      loss.backward()
      optimizer.step()
      train_losses.append(loss.item())
    val_losses = []
    model = model.eval()
    with torch.no_grad():
      for seq_true in val_dataset:
        seq_true = seq_true.to(device)
        seq_pred = model(seq_true)
        loss = criterion(seq_pred, seq_true)
        val_losses.append(loss.item())
    train_loss = np.mean(train_losses)
    val_loss   = np.mean(val_losses)
    history['train'].append(train_loss)
    history['val'].append(val_loss)
    if val_loss < best_loss:
      best_loss = val_loss
      best_model_wts = copy.deepcopy(model.state_dict())
    print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
  model.load_state_dict(best_model_wts)
  return model.eval(), history

model, history = train_model(
  model,
  train_dataset,
  val_dataset,
  n_epochs = 1 #150
)

# Saving model
# MODEL_PATH = 'model.pth'
# torch.save(model, MODEL_PATH)

# Choose a threshold for anomaly detection ------------------------------------
def predict(model, dataset):
  predictions, losses = [], []
  criterion = nn.L1Loss(reduction='sum').to(device)
  with torch.no_grad():
    model = model.eval()
    for seq_true in dataset:
      seq_true = seq_true.to(device)
      seq_pred = model(seq_true)
      loss = criterion(seq_pred, seq_true)
      predictions.append(seq_pred.cpu().numpy().flatten())
      losses.append(loss.item())
  return predictions, losses

import seaborn as sns
_, losses = predict(model, train_dataset)
sns.distplot(losses, bins=50, kde=True);


# Classify unseen examples as normal or anomaly -------------------------------
THRESHOLD = 26
predictions, pred_losses = predict(model, test_normal_dataset)
sns.distplot(pred_losses, bins=50, kde=True);

correct = sum(l <= THRESHOLD for l in pred_losses)
print(f'Correct normal predictions: {correct}/{len(test_normal_dataset)}')
# Correct normal predictions: 142/145

anomaly_dataset = test_anomaly_dataset[:len(test_normal_dataset)]
predictions, pred_losses = predict(model, anomaly_dataset)
sns.distplot(pred_losses, bins=50, kde=True);

correct = sum(l > THRESHOLD for l in pred_losses)
print(f'Correct anomaly predictions: {correct}/{len(anomaly_dataset)}')
