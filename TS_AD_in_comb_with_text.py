"""
@author: Mahdieh Khosravi
Detecting Anomalies in Time series through using textual information: 
Combining TS and text for Anomaly Detection
Anomaly detection with time series forecasting
"""
#=============================================================================#
#                            Import Libraries                                 #
#=============================================================================#
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import pandas as pd
import keras
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import BatchNormalization, Input, Embedding, Concatenate, Conv1D, MaxPooling1D, Flatten
from keras.models import Sequential, Model
import keras.backend as K
import statsmodels.formula.api as smf
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Prevent tf from allocating the entire GPU memory at once
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

#=============================================================================#
#                                  Parameters                                 #
#=============================================================================#
# Global parameters
NUM_LAGS = 10

# word embeddings parameters
GLOVE_DIR = "/glove.6B/"   # "/home/fmpr/datasets/glove.6B/"

MAX_SEQUENCE_LENGTH = 350 #600
MAX_NB_WORDS        = 600 #5000
EMBEDDING_DIM       = 300 #300

#=============================================================================#
#                                  Loading data                               #
#=============================================================================#
# ---------------------------- Load textual data ------------------------------
print( "loading text data..." )

txtdata = pd.read_csv("ticketset_text.tsv", sep="\t")
print(txtdata.head())

txtdata['start_time'] = pd.to_datetime(txtdata['start_time'], format='%Y-%m-%d %H:%M')
txtdata['date']       = txtdata['start_time'].dt.strftime("%Y-%m-%d")
txtdata               = txtdata[["date","start_time","title","url","description"]]

# -------------------------------- Load TS data -------------------------------
print( "loading time series ..." )

df = pd.read_csv("metric_meas_TS.csv")

df_sum = pd.DataFrame(df.groupby("date")["pickups"].sum())
df_sum["date"] = df_sum.index
df_sum.index   = pd.to_datetime(df_sum.index, format='%Y-%m-%d %H:%M')
df_sum["dow"]  = df_sum.index.weekday

# add text information
txt_col = np.zeros((len(df_sum)))

txt_desc_col = []
for i in range(len(df_sum)):
    if df_sum.iloc[i].date in txtdata["date"].values:
        txt_col[i] = 1
        txt_descr = ""
        for e in txtdata[txtdata.date == df_sum.iloc[i].date]["description"]:
            txt_descr += str(e) + " "
        txt_desc_col.append(txt_descr)

    else:
        txt_desc_col.append("None")

df_sum["txt"] = txt_col
df_sum["txt_desc"] = txt_desc_col
df_sum["txt_next_day"] = pd.Series(df_sum["txt"]).shift(-1)
df_sum["txt_next_day_desc"] = pd.Series(df_sum["txt_desc"]).shift(-1)


# keep only a part of data
START_YEAR = 2019
df_sum = df_sum.loc[df_sum.index.year >= START_YEAR]
df_sum.head()

df_sum["year"] = df_sum.index.year

trend_mean = df_sum[df_sum.index.year < 2020].groupby(["dow"]).mean()["pickups"]

#trend_std = df_sum.groupby(["year"]).std()["pickups"]
trend_std = df_sum["pickups"].std()

# build vectors with trend to remove and std
trend = []
std = []
for ix, row in df_sum.iterrows():
    trend.append(trend_mean[row.dow])
    #std.append(trend_std[row.year])
    std.append(trend_std)

df_sum["trend"] = trend
df_sum["std"]   = std

# detrend data
df_sum["detrended"] = (df_sum["pickups"] - df_sum["trend"]) / df_sum["std"]

#=============================================================================#
#                              Data Preparation                               #
#=============================================================================#
# ------------------------ Build lags and features ----------------------------
print( "building lags..." )

lags      = pd.concat([pd.Series(df_sum["detrended"]).shift(x) for x in range(0,NUM_LAGS)],axis=1).values   #.as_matrix()
tmdt      = pd.concat([pd.Series(df_sum["date"]).shift(x) for x in range(0,NUM_LAGS)],axis=1).values
txt_texts = df_sum["txt_next_day_desc"].values
preds     = pd.Series(df_sum["detrended"]).shift(-1).values
trends    = df_sum["trend"].values
stds      = df_sum["std"].values

lags      = lags[NUM_LAGS:-1,:]
tmdt      = tmdt[NUM_LAGS:-1,:]
txt_texts = txt_texts[NUM_LAGS:-1]

preds     = preds[NUM_LAGS:-1]
trends    = trends[NUM_LAGS:-1]
stds      = stds[NUM_LAGS:-1]

# ---------------------------------------- Train/test split
print( "loading train/val/test split..." )

i_train = len(lags)*0.4
i_val   = len(lags)*0.6
i_test  = -1 

lags_train      = lags[:i_train,:]    # time series lags
txt_texts_train = txt_texts[:i_train] # ticket text descriptions
y_train         = preds[:i_train]     # target values

lags_val        = lags[i_train:i_val,:]    # time series lags
txt_texts_val   = txt_texts[i_train:i_val] # ticket text descriptions
y_val           = preds[i_train:i_val]     # target values

lags_test      = lags[i_val:i_test,:]
txt_texts_test = txt_texts[i_val:i_test]
y_test         = preds[i_val:i_test]
trend_test     = trends[i_val:i_test]
std_test       = stds[i_val:i_test]

#=============================================================================#
#                           Function Definition                               #
#=============================================================================#
# --------------------------- Evaluation functions ----------------------------

def compute_error(trues, predicted):
    corr = np.corrcoef(predicted, trues)[0,1]
    mae = np.mean(np.abs(predicted - trues))
    rae = np.sum(np.abs(predicted - trues)) / np.sum(np.abs(trues - np.mean(trues)))
    rmse = np.sqrt(np.mean((predicted - trues)**2))
    rrse = np.sqrt(np.sum((predicted - trues)**2) / np.sum((trues - np.mean(trues))**2))
    mape = np.mean(np.abs((predicted - trues) / trues)) * 100
    r2 = max(0, 1 - np.sum((predicted - trues)**2) / np.sum((trues - np.mean(trues))**2))
    return corr, mae, rae, rmse, rrse, mape, r2


def compute_error_filtered(trues, predicted, filt):
    trues = trues[filt]
    predicted = predicted[filt]
    corr = np.corrcoef(predicted, trues)[0,1]
    mae = np.mean(np.abs(predicted - trues))
    mse = np.mean((predicted - trues)**2)
    rae = np.sum(np.abs(predicted - trues)) / np.sum(np.abs(trues - np.mean(trues)))
    rmse = np.sqrt(np.mean((predicted - trues)**2))
    r2 = max(0, 1 - np.sum((trues-predicted)**2) / np.sum((trues - np.mean(trues))**2))
    return corr, mae, mse, rae, rmse, r2

# ------------------------ Anomaly Detection (AD) -----------------------------
def detect_anomalies(df,window):
    df.replace([np.inf, -np.inf], np.NaN, inplace=True)
    df.fillna(0,inplace=True)
    df['error'] = df['actuals']-df['predicted']
    df['percentage_change'] = ((df['actuals'] - df['predicted']) / df['actuals']) * 100
    df['meanval']   = df['error'].rolling(window=window).mean()
    df['deviation'] = df['error'].rolling(window=window).std()
    df['-3s']  = df['meanval'] - (2 * df['deviation'])
    df['3s']   = df['meanval'] + (2 * df['deviation'])
    df['-2s']  = df['meanval'] - (1.75 * df['deviation'])
    df['2s']   = df['meanval'] + (1.75 * df['deviation'])
    df['-1s']  = df['meanval'] - (1.5 * df['deviation'])
    df['1s']   = df['meanval'] + (1.5 * df['deviation'])
    cut_list   = df[['error', '-3s', '-2s', '-1s', 'meanval', '1s', '2s', '3s']]
    cut_values = cut_list.values
    cut_sort   = np.sort(cut_values)
    df['impact'] = [(lambda x: np.where(cut_sort == df['error'][x])[1][0])(x) for x in
                               range(len(df['error']))]
    severity = {0: 3, 1: 2, 2: 1, 3: 0, 4: 0, 5: 1, 6: 2, 7: 3}
    region   = {0: "NEGATIVE", 1: "NEGATIVE", 2: "NEGATIVE", 3: "NEGATIVE", 4: "POSITIVE", 5: "POSITIVE", 6: "POSITIVE",
              7: "POSITIVE"}
    df['color']  = df['impact'].map(severity)
    df['region'] = df['impact'].map(region)
    df['anomaly_points'] = np.where(df['color'] == 3, df['error'], np.nan)
    df = df.sort_values(by='date', ascending=False)
    df.date = pd.to_datetime(df['date'].astype(str), format="%Y-%m-%d")

    return df

# -------------------------- Output files -------------------------------------

if not os.path.exists("results_mae.txt"):
    fw_mae = open("results_mae.txt", "a")
    fw_mae.write("LR L,LR L+W,LR L+W+E,LR L+W+E+LF,LR L+W+E+LF+EL,")
    fw_mae.write("MLP L,MLP L+W,MLP L+W+E,MLP L+W+E+LF,MLP L+W+E+LF+EL,")
    fw_mae.write("MLP L+W+E+LF+ET,MLP L+W+E+LF+EL+ET\n")
    fw_rae = open("results_rae.txt", "a")
    fw_rae.write("LR L,LR L+W,LR L+W+E,LR L+W+E+LF,LR L+W+E+LF+EL,")
    fw_rae.write("MLP L,MLP L+W,MLP L+W+E,MLP L+W+E+LF,MLP L+W+E+LF+EL,")
    fw_rae.write("MLP L+W+E+LF+ET,MLP L+W+E+LF+EL+ET\n")
    fw_rmse = open("results_rmse.txt", "a")
    fw_rmse.write("LR L,LR L+W,LR L+W+E,LR L+W+E+LF,LR L+W+E+LF+EL,")
    fw_rmse.write("MLP L,MLP L+W,MLP L+W+E,MLP L+W+E+LF,MLP L+W+E+LF+EL,")
    fw_rmse.write("MLP L+W+E+LF+ET,MLP L+W+E+LF+EL+ET\n")
    fw_rrse = open("results_rrse.txt", "a")
    fw_rrse.write("LR L,LR L+W,LR L+W+E,LR L+W+E+LF,LR L+W+E+LF+EL,")
    fw_rrse.write("MLP L,MLP L+W,MLP L+W+E,MLP L+W+E+LF,MLP L+W+E+LF+EL,")
    fw_rrse.write("MLP L+W+E+LF+ET,MLP L+W+E+LF+EL+ET\n")
    fw_mape = open("results_mape.txt", "a")
    fw_mape.write("LR L,LR L+W,LR L+W+E,LR L+W+E+LF,LR L+W+E+LF+EL,")
    fw_mape.write("MLP L,MLP L+W,MLP L+W+E,MLP L+W+E+LF,MLP L+W+E+LF+EL,")
    fw_mape.write("MLP L+W+E+LF+ET,MLP L+W+E+LF+EL+ET\n")
    fw_r2 = open("results_r2.txt", "a")
    fw_r2.write("LR L,LR L+W,LR L+W+E,LR L+W+E+LF,LR L+W+E+LF+EL,")
    fw_r2.write("MLP L,MLP L+W,MLP L+W+E,MLP L+W+E+LF,MLP L+W+E+LF+EL,")
    fw_r2.write("MLP L+W+E+LF+ET,MLP L+W+E+LF+EL+ET\n")
else:
    fw_mae = open("results_mae.txt", "a")
    fw_rae = open("results_rae.txt", "a")
    fw_rmse = open("results_rmse.txt", "a")
    fw_rrse = open("results_rrse.txt", "a")
    fw_mape = open("results_mape.txt", "a")
    fw_r2 = open("results_r2.txt", "a")

#=============================================================================#
#                         Time Series Forecasting                             #
#=============================================================================#

# ------------------------------ LSTM (just lags) -----------------------------

def build_model(num_inputs, num_lags, num_preds):
    input_lags = Input(shape=(num_lags,1))
    
    x = input_lags
    
    x = LSTM(20, 
             kernel_regularizer=keras.regularizers.l2(0.1), 
             recurrent_regularizer=keras.regularizers.l2(0.1), 
             return_sequences=False)(x)
    #x = Activation("relu")(x)
    #x = Flatten()(x)
    x = BatchNormalization()(x)
    #x = Dropout(0.2)(x)
    #x = BatchNormalization()(x)
    
    #preds = Dense(units=num_preds)(x)
    preds = Dense(units=num_preds, kernel_regularizer=keras.regularizers.l2(0.01))(x)
    
    model = Model(input_lags, preds)
    
    #rmsp = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss="mse", optimizer="rmsprop")
    
    return model, input_lags, preds


print( "\nrunning LSTM with just lags..." )

# checkpoint best model
checkpoint = ModelCheckpoint("weights.best.hdf5", monitor='val_loss', verbose=0, save_best_only=True, mode='min')

model, input_lags, preds = build_model(1, NUM_LAGS, 1)
model.fit(
    lags_train[:,:,np.newaxis],
    y_train,
    batch_size = 64,
    epochs     = 100,    #1000
    validation_data = (lags_val[:,:,np.newaxis], y_val),
    callbacks       = [checkpoint],
    verbose = 0)   

print( "Total number of iterations:  " , len(model.history.history["loss"]))
print( "Best loss at iteratation:    " , np.argmin(model.history.history["loss"]), "   Best:", np.min(model.history.history["loss"]))
print( "Best val_loss at iteratation:" , np.argmin(model.history.history["val_loss"]), "   Best:", np.min(model.history.history["val_loss"]))

# load weights
model.load_weights("weights.best.hdf5")

# make predictions
preds_lstm = model.predict(np.concatenate([lags_test[:,:,np.newaxis]], axis=1))
preds_lstm = preds_lstm[:,0] * std_test + trend_test
y_true     = y_test * std_test + trend_test
corr, mae, rae, rmse, rrse, mape, r2 = compute_error(y_true, preds_lstm)
#print( "MAE:  %.3f\tRMSE: %.3f\tR2:   %.3f"  % (mae, rmse, r2))
print( "MAE:  %.3f\tRMSE: %.3f\tR2:   %.1f"  % (mae, rmse, r2*100))
fw_mae.write("%.3f," % (mae,))
fw_rae.write("%.3f," % (rae,))
fw_rmse.write("%.3f," % (rmse,))
fw_rrse.write("%.3f," % (rrse,))
fw_mape.write("%.3f," % (mape,))
fw_r2.write("%.3f," % (r2,))


# -------------------------- LSTM with lags + TEXT ----------------------------

print( "\npreparing word embeddings for NNs with text..." )

# Build index mapping words in the embeddings set to their embedding vector
embeddings_index = {}
f = open('glove.6B.%dd.txt' % (EMBEDDING_DIM,))   #GLOVE_DIR + 
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

# Vectorize the text samples into a 2D integer tensor and pad sequences
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(txt_texts)
sequences_train = tokenizer.texts_to_sequences(txt_texts_train)
sequences_val   = tokenizer.texts_to_sequences(txt_texts_val)
sequences_test  = tokenizer.texts_to_sequences(txt_texts_test)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data_train = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH)
data_val = pad_sequences(sequences_val, maxlen=MAX_SEQUENCE_LENGTH)
data_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of train tensor:', data_train.shape)
print('Shape of val tensor:', data_val.shape)
print('Shape of test tensor:', data_test.shape)

# Prepare embedding matrix
print('Preparing embedding matrix.')
num_words = min(MAX_NB_WORDS, len(word_index)+1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    #print i
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


def build_model_text(num_inputs, num_lags, num_feat, num_preds):
    input_lags = Input(shape=(num_lags,1))
    
    x_lags = input_lags    
    #x_lags = BatchNormalization()(x_lags)
    x_lags = LSTM(20, 
             kernel_regularizer=keras.regularizers.l2(0.1), 
             recurrent_regularizer=keras.regularizers.l2(0.1), 
             return_sequences=False)(x_lags)
    #x_lags = Activation("relu")(x_lags)
    #x_lags = Flatten()(x_lags)
    x_lags = BatchNormalization()(x_lags)
    #x_lags = Dropout(0.2)(x_lags)
    #x_lags = BatchNormalization()(x_lags)

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(50, 3, activation='relu')(embedded_sequences)
    x = MaxPooling1D(3)(x)
    x = Dropout(rate=0.5)(x)
    x = Conv1D(30, 3, activation='relu')(x)
    x = MaxPooling1D(3)(x)
    x = Dropout(rate=0.5)(x)
    x = Conv1D(30, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    #x = Dropout(0.1)(x)
    #x = Conv1D(50, 5, activation='relu')(x)
    #x = MaxPooling1D(5)(x)
    text_embedding = Flatten()(x)
    text_embedding = Dropout(rate=0.5)(text_embedding)
    #text_embedding = Dense(units=100, activation='relu')(text_embedding)
    #text_embedding = Dropout(0.5)(text_embedding)

    feat = Concatenate(axis=1)([x_lags, text_embedding])

    #preds = Dense(units=num_preds)(feat)
    preds = Dense(units=num_preds, kernel_regularizer=keras.regularizers.l2(0.01))(feat)
    # preds = Activation("linear")(preds)
    
    model = Model([input_lags, sequence_input], preds)
    
    rmsp = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    #model.compile(loss="mse", optimizer=rmsp)
    model.compile(loss="mse", optimizer="rmsprop")
    
    return model, input_lags, preds


print( "\nrunning LSTM with lags + text..." )

# checkpoint best model
checkpoint = ModelCheckpoint("weights.best.hdf5", monitor='val_loss', verbose=0, save_best_only=True, mode='min')

# fit model to the mean
model, input_lags, preds = build_model_text(1, NUM_LAGS, 1, 1)    # 3rd arg. should be....!!!!
model.fit(

    [lags_train[:,:,np.newaxis], data_train],
    y_train,
    batch_size = 64,
    epochs     = 100,   #1000

    validation_data=([lags_val[:,:,np.newaxis], data_val], y_val),
    callbacks=[checkpoint],
    verbose=0)    

print( "Total number of iterations:  " , len(model.history.history["loss"]))
print( "Best loss at iteratation:    " , np.argmin(model.history.history["loss"]), "   Best:", np.min(model.history.history["loss"]))
print( "Best val_loss at iteratation:" , np.argmin(model.history.history["val_loss"]), "   Best:", np.min(model.history.history["val_loss"]))

# load weights
model.load_weights("weights.best.hdf5")

print(model.evaluate([lags_test[:,:,np.newaxis], data_test], 
                      y_test, verbose=2))

# make predictions
preds_lstm = model.predict([lags_test[:,:,np.newaxis], data_test])
preds_lstm = preds_lstm[:,0] * std_test + trend_test
corr, mae, rae, rmse, rrse, mape, r2 = compute_error(y_true, preds_lstm)
#print( "MAE:  %.3f\tRMSE: %.3f\tR2:   %.3f"  % (mae, rmse, r2))
print( "MAE:  %.3f\tRMSE: %.3f\tR2:   %.1f"  % (mae, rmse, r2*100))
fw_mae.write("%.3f," % (mae,))
fw_rae.write("%.3f," % (rae,))
fw_rmse.write("%.3f," % (rmse,))
fw_rrse.write("%.3f," % (rrse,))
fw_mape.write("%.3f," % (mape,))
fw_r2.write("%.3f," % (r2,))



# close output files
fw_mae.write("\n")
fw_rae.write("\n")
fw_rmse.write("\n")
fw_rrse.write("\n")
fw_mape.write("\n")
fw_r2.write("\n")
fw_mae.close()
fw_rae.close()
fw_rmse.close()
fw_rrse.close()
fw_mape.close()
fw_r2.close()

#=============================================================================#
#                             Anomaly Detection                               #
#=============================================================================#
pred_df              = pd.DataFrame()
pred_df['date']      = tmdt[i_val:i_test][:,0]
pred_df['actuals']   = y_true
pred_df['predicted'] = preds_lstm    #preds
pred_df.reset_index(inplace=True)
del pred_df['index']
print(pred_df.head())

anomaly_df = detect_anomalies(pred_df,7)    
anomaly_df.reset_index(inplace=True)
del anomaly_df['index']
print(anomaly_df.head())
