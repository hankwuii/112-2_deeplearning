import torch
from torch.nn import Module, Sequential
from torch.nn import Linear, ReLU, Sigmoid
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import pandas as pd
from Neural_Network import Neural_Network
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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

            self.fc_output = Sequential(
                Linear(layer_3_neurons, num_classes),
                Sigmoid()
            )               

                        

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
    data = pd.read_csv(data_file)
    
    target_column = 'AboveMedianPrice'
    
    X = data.drop(target_column, axis=1).values
    y = data[target_column].values.reshape(-1, 1)
    # to PyTorch Tensor
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return TensorDataset(X_tensor, y_tensor)

# load dataset
data_path = 'data/NN_Classification/housepricedata.csv'
batch_size = 64
num_epochs = 200
learning_rate = 0.001

dataset = process_data(data_path)
X_train, X_test, y_train, y_test = train_test_split(dataset.tensors[0], dataset.tensors[1], test_size=0.2, random_state=42)

training_dataset = TensorDataset(X_train, y_train)
training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

testing_dataset = TensorDataset(X_test, y_test)
testing_dataloader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=True)


num_features = training_dataset[0][0].size(0)
model = Neural_Network(num_features, num_classes=1)

criterion = torch.nn.BCELoss()
optimizer = Adam(model.parameters(), lr=learning_rate)


train_losses = []
test_losses = []


# 訓練模型
for epoch in range(num_epochs):
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    for inputs, labels in training_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
        train_correct += (torch.round(outputs) == labels).sum().item()
        train_total += labels.size(0)
    train_loss /= len(training_dataloader.dataset)
    train_losses.append(train_loss)

    test_loss = 0.0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, labels in testing_dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            test_correct += (torch.round(outputs) == labels).sum().item()
            test_total += labels.size(0)
    test_loss /= len(testing_dataloader.dataset)
    test_losses.append(test_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

# 繪製損失圖
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), test_losses, label='Testing Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Testing Loss')
plt.legend()


plt.show()