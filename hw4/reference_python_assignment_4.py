import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 讀取訓練數據
data_train = pd.read_csv('C:/Users/User/Desktop/Deep learning/20240429/Google_Stock_Price/Google_Stock_Price_Train.csv', dtype={'Close': np.float64})
x_train_data = data_train[['Open', 'High', 'Low']].values
y_train_data = data_train[['Close']].values

# 特徵縮放
scaler_x_train = StandardScaler()
scaler_y_train = StandardScaler()
x_train_scaled_data = scaler_x_train.fit_transform(x_train_data)
y_train_scaled_data = scaler_y_train.fit_transform(y_train_data)

# 分割訓練集和驗證集
X_train_data, X_valid_data, y_train_data, y_valid_data = train_test_split(x_train_scaled_data, y_train_scaled_data, test_size=0.2, random_state=45)

# 定義神經網絡模型
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(3, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 6)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc4(x)
        return x

# 初始化神經網絡模型
model_nn = NeuralNet()

# 定義損失函數和優化器
loss_function = nn.MSELoss()
optimizer_nn = optim.Adam(model_nn.parameters(), lr=0.005)

# 訓練神經網絡模型
num_epochs_nn = 600
train_losses_nn = []
valid_losses_nn = []

for epoch in range(num_epochs_nn):
    model_nn.train()
    optimizer_nn.zero_grad()
    outputs_nn = model_nn(torch.tensor(X_train_data, dtype=torch.float32))
    loss_nn = loss_function(outputs_nn, torch.tensor(y_train_data, dtype=torch.float32))
    loss_nn.backward()
    optimizer_nn.step()
    train_losses_nn.append(loss_nn.item())

    model_nn.eval()
    with torch.no_grad():
        valid_outputs_nn = model_nn(torch.tensor(X_valid_data, dtype=torch.float32))
        valid_loss_nn = loss_function(valid_outputs_nn, torch.tensor(y_valid_data, dtype=torch.float32))
        valid_losses_nn.append(valid_loss_nn.item())

# 繪製訓練和驗證損失曲線
plt.figure()
plt.plot(range(1, num_epochs_nn + 1), train_losses_nn, 'r', label='Training loss')
plt.plot(range(1, num_epochs_nn + 1), valid_losses_nn, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 讀取測試數據
data_test = pd.read_csv("C:/Users/User/Desktop/Deep learning/20240429/Google_Stock_Price/Google_Stock_Price_Test.csv", dtype={'Close': np.float64})
x_test_data = data_test[['Open', 'High', 'Low']].values

# 對測試數據進行特徵縮放
x_test_scaled_data = scaler_x_train.transform(x_test_data)

# 使用神經網絡模型預測測試數據的股價
model_nn.eval()
with torch.no_grad():
    y_tensor_nn = model_nn(torch.tensor(x_test_scaled_data, dtype=torch.float32))

# 將預測結果轉換回原始尺度
y_price_nn = scaler_y_train.inverse_transform(y_tensor_nn.numpy())

# 將神經網絡模型的輸出作為特徵，與原始特徵合併
train_features_nn = model_nn(torch.tensor(x_train_scaled_data, dtype=torch.float32)).detach().numpy()
X_train_combined_nn = np.concatenate((x_train_scaled_data, train_features_nn), axis=1)
test_features_nn = model_nn(torch.tensor(x_test_scaled_data, dtype=torch.float32)).detach().numpy()
X_test_combined_nn = np.concatenate((x_test_scaled_data, test_features_nn), axis=1)

# 使用隨機森林模型
rf_model_nn = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model_nn.fit(X_train_combined_nn, y_train_scaled_data.squeeze())
y_pred_combined_rf_nn = rf_model_nn.predict(X_test_combined_nn)

# 將隨機森林模型的預測結果轉換回原始尺度
y_pred_original_rf_nn = scaler_y_train.inverse_transform(y_pred_combined_rf_nn.reshape(-1, 1))

# 繪製結果
plt.figure()
plt.plot(data_test['Close'], color='g', label='Target')
plt.plot(y_pred_original_rf_nn, color='b', label='Predict (Random Forest)')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
plt.show()
