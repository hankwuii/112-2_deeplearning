# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 10:51:42 2020

@author: USER
"""


import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.layers import Dropout
from keras import regularizers
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import numpy
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

#在此將houseprice的檔案讀進去df中
#注意注意，在此如果檔案路徑不同會跳錯，所以記得把檔案位置改成data所放的位置
#例如放在D槽，就是df = pd.read_csv('D:\\housepricedata.csv')
#記得在讀檔案的時候要用手打，不然跳錯機率極高
df = pd.read_csv('D:\\intuitive-deep-learning-master\\Part 1_ Predicting House Prices\\housepricedata.csv')

#將df轉int成只有數字的模式，將檔案的英文轉為0~10
dataset = df.values

#將前十項命名為X
X = dataset[:,0:10]

#第十一項為Y
Y = dataset[:,10]

#將Y的部分轉LabelEncoder使其變為0or1
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

#將整數轉換為虛擬變量---One-Hot Encoding
dummy_y = np_utils.to_categorical(encoded_Y)

#把X做min_max_scaler___在此他是第一個LotArea整行對自己做min_max，然後第二個OverallQual整行對自己做min_max....~最後一個變數
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)

#將資料分為訓練集以及驗證集
X_train, X_val_and_test, dummy_y, Y_val_and_test = train_test_split(X_scale, dummy_y, test_size=0.3)

#再將驗證集分為驗證集以及測試集
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

#可以看各自的大小
print(X_train.shape, X_val.shape, X_test.shape, dummy_y.shape, Y_val.shape, Y_test.shape)

#建立模型
model = Sequential()
#在此他的神經元units可自行去做調整，INPUT_DIM為10是因為x的大小為10
model.add(Dense(units=32, input_dim=10, kernel_initializer='normal', activation='relu')) 
model.add(Dense(units=32, kernel_initializer='normal', activation='relu'))
model.add(Dense(units=20, kernel_initializer='normal', activation='relu'))
model.add(Dense(units=2, kernel_initializer='normal', activation='softmax')) 


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#model.fit中前面擺訓練集,後方dummy_y為標籤label，筆數大小一樣才可以進行訓練
print(X_train.shape,dummy_y.shape)
print(X_val.shape, Y_val.shape)
hist = model.fit(X_train, dummy_y,
          batch_size=32, epochs=100,
          validation_data=(X_val, Y_val))

#在這裡我們可以看到模型的accuracy以及Loss
#在此的秀圖為模型整體的accuracy為多少是否有上升的趨勢
plt.plot(hist.history['acc'], 'b', label='train')
plt.plot(hist.history['val_acc'], 'r', label='valid')
plt.legend()
plt.grid(True)
plt.title('accuracy')
plt.show()
model.evaluate(X_test, Y_test)[1]

#在此的秀圖為模型整體的loss為多少是否有下降穩定趨勢
plt.plot(hist.history['loss'], 'b', label='train')
plt.plot(hist.history['val_loss'], 'r', label='valid')
plt.legend()
plt.grid(True)
plt.title('loss')
plt.show()
model.evaluate(X_test, Y_test)[1]



#__________在這裡我們可以看到把神經元加到1000以後的acc以及Loss
#建立模型
model2 = Sequential()

model2.add(Dense(units=1000, input_dim=10, kernel_initializer='normal', activation='relu')) 
model2.add(Dense(units=1000, kernel_initializer='normal', activation='relu'))
model2.add(Dense(units=1000, kernel_initializer='normal', activation='relu'))
model2.add(Dense(units=2, kernel_initializer='normal', activation='softmax')) 

model2.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
hist_2 = model2.fit(X_train, dummy_y,
          batch_size=32, epochs=50,
          validation_data=(X_val, Y_val))

plt.plot(hist_2.history['acc'], 'b', label='train')
plt.plot(hist_2.history['val_acc'], 'r', label='valid')
plt.legend()
plt.grid(True)
plt.title('accuracy')
plt.show()
model2.evaluate(X_test, Y_test)[1]


########可以多去了解為何loss長那個樣子
plt.plot(hist_2.history['loss'])
plt.plot(hist_2.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()


#__________在這裡我們可以看到把神經元加到1000以及多了dropout的acc以及Loss
########可以多去了解為何loss長那個樣子
model_3 = Sequential([
    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(10,)),
    Dropout(0.3),
    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(0.01)),
])

model_3.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
hist_3 = model_3.fit(X_train, dummy_y,
          batch_size=32, epochs=100,
          validation_data=(X_val, Y_val))


plt.plot(hist_3.history['acc'], 'b', label='train')
plt.plot(hist_3.history['val_acc'], 'r', label='valid')
plt.legend()
plt.grid(True)
plt.title('accuracy')
plt.show()
model_3.evaluate(X_test, Y_test)[1]


plt.plot(hist_3.history['loss'])
plt.plot(hist_3.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.ylim(top=1.2, bottom=0)
plt.show()