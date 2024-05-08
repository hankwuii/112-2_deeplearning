# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 00:34:42 2020

@author: 江嘉晉
"""


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt#繪圖套件
from sklearn import preprocessing#預處理套件
from sklearn.preprocessing import MinMaxScaler#標準化其中一種方法
from sklearn.model_selection import train_test_split#切割訓練集以及驗證集用
import tensorflow as tf#深度學習套件
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense#建構NN模型用
from tensorflow.keras.layers import SimpleRNN#建構RNN模型用
from tensorflow.keras.layers import Dropout
import os
import warnings 
#忽略警告的提示
warnings.filterwarnings("ignore")


# 寫入當前目錄的所有結果都將保存為輸出
#讀檔
dataset_train = pd.read_csv('D:\\Google_Stock_Price\\Google_Stock_Price_Train.csv', dtype={'Close':np.float64})

#顯示前面5筆資料
dataset_train.head()

#'Open','High','Low'作為訓練模型的因子
x_train = dataset_train.loc[:,["Open",'High','Low']].values#(.values)將資料表型態資料轉成數值陣列的型態
#"Close"當作預測目標
targets = dataset_train.loc[:,["Close"]].values

#將train資料標準化
scaler = preprocessing.StandardScaler()#最小最大值標準化
#X_train = scaler.fit_transform(x_train)
X_train = []
for i in range(0, x_train.shape[1]):
    X_train.append(scaler.fit_transform(x_train[:,i].reshape(-1,1)))
    x_train = np.concatenate((x_train, X_train[i]), axis = 1)

#選取x_train標準化後的欄位為訓練模型feature
features = x_train[:,3:x_train.shape[1]]
#標準化label價格
targets = scaler.fit_transform(targets)

#切割訓練集以及驗證集
X_train, X_valid, y_train, y_valid = train_test_split(features, 
                                                    targets, 
                                                    test_size=0.20,
                                                    random_state=45)

#建構NN模型
model = Sequential()
# Add Input layer, 隱藏層(hidden layer)
model.add(Dense(units=45, kernel_initializer='normal',input_shape = (X_train.shape[1], ), activation='relu')) 
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=128, activation='relu')) 
model.add(Dense(units=256, activation='tanh'))
# Add output layer
model.add(Dense(units=1, activation='linear'))

model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['mse'])

#開始訓練模型
history = model.fit(X_train, y_train, epochs = 30, batch_size = 32, validation_data=(X_valid, y_valid), shuffle=True)

history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = history.epoch

#圖示化loss收斂情況
plt.figure(figsize=(12,9))
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss', size=20)
plt.xlabel('Epochs', size=20)
plt.ylabel('Loss', size=20)
plt.legend(prop={'size': 20})
plt.show()

#讀取測試集
dataset_test = pd.read_csv("D:\\Google_Stock_Price\\Google_Stock_Price_Test.csv", dtype={'Close':np.float64})
dataset_test.head()

#'Open','High','Low'當作預測因子
x_test = dataset_test.loc[:,['Open','High','Low']].values
real_stock_price = dataset_test.loc[:, ["Close"]].values
real_stock_price.shape
#x_test_scaled = scaler.fit_transform(X_test)
X_test = []
for i in range(0, x_test.shape[1]):
    X_test.append(scaler.fit_transform(x_test[:,i].reshape(-1,1)))
    x_test = np.concatenate((x_test, X_test[i]), axis = 1)

#選取x_test標準化後的欄位為預測因子
X_test = x_test[:,3:x_test.shape[1]]


#預測
predicted_stock_price = model.predict(X_test)
#將預測標準化後的值轉回實際價格
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

#圖示化真實股價與預測股價
plt.plot(real_stock_price, color = 'green', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'red', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

#儲存model權重
model.save('my_model.h5')









