# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 14:56:35 2020

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

#把X做min_max_scaler___在此他是第一個LotArea整行對自己做min_max，然後第二個OverallQual整行對自己做min_max....~最後一個變數
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)

#將資料分為訓練集以及驗證集
X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)

#再將驗證集分為驗證集以及測試集
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)


#可以看各自的大小
print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)

#建立模型___在此他的神經元units可自行去做調整，INPUT_DIM為10是因為x的大小為10
model = Sequential([
    Dense(32, activation='relu', input_shape=(10,)),
    Dense(32, activation='relu'),
    Dense(20, activation='relu'),
    Dense(1, activation='sigmoid'),
])

model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#model.fit中前面擺訓練集,後方Y_train為標籤label，筆數大小一樣才可以進行訓練
print(X_train.shape,Y_train.shape)
hist = model.fit(X_train, Y_train,
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
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

#__________在這裡我們可以看到把神經元加到1000以後的acc以及Loss
#建立模型
model_2 = Sequential([
    Dense(1000, activation='relu', input_shape=(10,)),
    Dense(1000, activation='relu'),
    Dense(1000, activation='relu'),
    Dense(1000, activation='relu'),
    Dense(1, activation='sigmoid'),
])
model_2.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
hist_2 = model_2.fit(X_train, Y_train,
          batch_size=32, epochs=50,
          validation_data=(X_val, Y_val))


plt.plot(hist_2.history['acc'], 'b', label='train')
plt.plot(hist_2.history['val_acc'], 'r', label='valid')
plt.legend()
plt.grid(True)
plt.title('accuracy')
plt.show()
model_2.evaluate(X_test, Y_test)[1]


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
########可以多去了解為何loss長那個樣子
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
    Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)),
])

model_3.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
hist_3 = model_3.fit(X_train, Y_train,
          batch_size=32, epochs=100,
          validation_data=(X_val, Y_val))


plt.plot(hist_3.history['acc'], 'b', label='train')
plt.plot(hist_3.history['val_acc'], 'r', label='valid')
plt.legend()
plt.grid(True)
plt.title('accuracy')
plt.show()
model_3.evaluate(X_test, Y_test)[1]


########可以多去了解為何loss長那個樣子
plt.plot(hist_3.history['loss'])
plt.plot(hist_3.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.ylim(top=1.2, bottom=0)
plt.show()