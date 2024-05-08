
import torch
from torch.nn import Module, Sequential
from torch.nn import Linear, ReLU, Sigmoid

from typing import List

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
    