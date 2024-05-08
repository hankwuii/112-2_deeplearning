import torch
from torch.nn import MSELoss
from torch.nn import Module, Sequential
from torch.nn import Linear, ReLU, Sigmoid
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

from typing import List
import pandas as pd

class Neural_Network(Module):
        def __init__(self, num_features: int, num_classes: int):
            super(Neural_Network, self).__init__()

            #------------------------------------#
            #          Hyperparameters           #
            #------------------------------------#

            # num_features : Number of input features
            # num_classes : Number of output features

            layer_1_neurons = 10
            layer_2_neurons = 20
            layer_3_neurons = 30

            #------------------------------------#
            #               Layers               #
            #------------------------------------#

            self.fc_input = Sequential(
                Linear(num_features, layer_1_neurons),
                ReLU()
            )

            self.fc_l1 = Sequential(
                Linear(layer_1_neurons, layer_2_neurons),
                ReLU()
            )

            self.fc_l2 = Sequential(
                Linear(layer_2_neurons, layer_3_neurons),
                ReLU()
            )


            # ---------------------------------- 問題!!!!! --------------------------------- #
            # 用Sigmoid()去做regression loss會抖動

            # self.fc_output = Sequential(
            #     Linear(layer_3_neurons, num_classes),
            #     Sigmoid()
            # )               
            self.fc_output = Linear(layer_3_neurons, num_classes)
                
            

        def forward(self, x):

            #------------------------------------#
            #             Pipeline               #
            #------------------------------------#

            x = self.fc_input(x)
            x = self.fc_l1(x)
            x = self.fc_l2(x)
            output = self.fc_output(x)

            return output




# ----------------------------------- 資料預處理 ---------------------------------- #
def process_data(data_file):
    # 讀取資料
    data = pd.read_csv(data_file)

    # 處理日期
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date'] = data['Date'].dt.dayofyear

    # 處理其他數字特徵
    data['Volume'] = data['Volume'].str.replace(',', '').astype(float)

    # 分割特徵和目標變數
    X = data[['Date', 'Open', 'High', 'Low']].values  # 只取 Date、Open、High、Low 欄位作為特徵
    y = data['Close'].values.reshape(-1, 1)  # 取 Close 欄位作為目標變數

    # 最小-最大標準化
    scaler = MinMaxScaler()
    y_scaled = scaler.fit_transform(y)

    # 將資料轉換為 PyTorch Tensor
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)

    return dataset



# --------------------------------- Load data -------------------------------- #
training_data_path = 'data/NN_Regression/Google_Stock_Price/Google_Stock_Price_Train.csv'
testing_data_path = 'data/NN_Regression/Google_Stock_Price/Google_Stock_Price_Test.csv'
target_column = 'Close'
batch_size = 64
num_epochs = 200
learning_rate = 0.001


training_dataset = process_data(training_data_path)
training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

testing_dataset = process_data(testing_data_path)
testing_dataloader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=True)



# ------------------------------- Initialize NN ------------------------------ #
num_features = training_dataset[0][0].size(0)   # input size, should be 4
num_classes = training_dataset[0][1].size(0)    # output size, should be 1

# print(f'input: {num_features}, outputs: {num_classes}')


model = Neural_Network(num_features, num_classes)



# -------------------- Define loss function and optimizer -------------------- #
criterion = MSELoss()
optimizer = Adam(model.parameters(), lr=learning_rate)


train_losses = []
test_losses = []


# --------------------------------- Training --------------------------------- #
for epoch in range(num_epochs):
    model.train()  # training mode
    for inputs, labels in training_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    train_losses.append(loss.item())

    # Testing
    model.eval()  # evaluation mode
    with torch.no_grad():
        total_loss = 0.0
        num_samples = 0
        for inputs, labels in testing_dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            num_samples += inputs.size(0)
        average_loss = total_loss / num_samples
        test_losses.append(average_loss)

    # Print the training and testing losses for this epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}')

# Plot
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Testing Losses')
plt.legend()
plt.show()


