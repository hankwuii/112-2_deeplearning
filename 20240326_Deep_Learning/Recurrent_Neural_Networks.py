
import torch
from torch.nn import Module, Sequential
from torch.nn import Linear, RNN, GRU, LSTM, Sigmoid

#---------------------------------------------------------------------------------------------------#
#                                  Elman Recurrent Neural Network                                   #
#---------------------------------------------------------------------------------------------------#

class Recurrent_Neural_Network(Module):
        def __init__(self, num_features: int, num_classes: int):
            super(Recurrent_Neural_Network, self).__init__()

            #------------------------------------#
            #          Hyperparameters           #
            #------------------------------------#

            # num_features : Number of input features
            # num_classes : Number of output features

            hidden_dim_n = 8
            self.hidden_dim_n = hidden_dim_n

            layers_n = 2
            self.layers_n = layers_n

            rnn_activation_function = 'tanh'

            #------------------------------------#
            #               Layers               #
            #------------------------------------#

            self.rnn = RNN(num_features, hidden_dim_n, layers_n, rnn_activation_function, batch_first = True)
            self.fc_output = Sequential(
                 Linear(hidden_dim_n, num_classes),
                 Sigmoid()
            )

        def forward(self, x):

            #------------------------------------#
            #             Pipeline               #
            #------------------------------------#

            batch_size = x.size[0]

            hidden = self.init_hidden(batch_size)

            out, hidden = self.rnn(x, hidden)

            out = out.contiguous().view(-1, self.hidden_dim)

            out = self.fc(out)
        
            return out, hidden
    
        def init_hidden(self, batch_size):
            # Generate first input tensor
            hidden = torch.zeros(self.layers_n, batch_size, self.hidden_dim_n)
            return hidden
        
#---------------------------------------------------------------------------------------------------#
#                                Gated Recurrent Unit Neural Network                                #   
#---------------------------------------------------------------------------------------------------#

class GRU_Network(Module):
        def __init__(self, num_features: int, num_classes: int):
            super(GRU_Network, self).__init__()

            #------------------------------------#
            #          Hyperparameters           #
            #------------------------------------#

            # num_features : Number of input features
            # num_classes : Number of output features

            hidden_dim_n = 8
            self.hidden_dim_n = hidden_dim_n

            layers_n = 2
            self.layers_n = layers_n

            #------------------------------------#
            #               Layers               #
            #------------------------------------#

            self.rnn = GRU(num_features, hidden_dim_n, layers_n, batch_first = True)
            self.fc_output = Sequential(
                 Linear(hidden_dim_n, num_classes),
                 Sigmoid()
            )

        def forward(self, x):

            #------------------------------------#
            #             Pipeline               #
            #------------------------------------#

            batch_size = x.size[0]

            hidden = self.init_hidden(batch_size)

            out, hidden = self.rnn(x, hidden)

            out = out.contiguous().view(-1, self.hidden_dim)

            out = self.fc(out)
        
            return out, hidden
    
        def init_hidden(self, batch_size):
            # Generate first input tensor
            hidden = torch.zeros(self.layers_n, batch_size, self.hidden_dim_n)
            return hidden
        
#---------------------------------------------------------------------------------------------------#
#                          Long Short-Term Memory Recurrent Neural Network                          #      
#---------------------------------------------------------------------------------------------------#

class LSTM_Network(Module):
        def __init__(self, num_features: int, num_classes: int):
            super(LSTM_Network, self).__init__()

            #------------------------------------#
            #          Hyperparameters           #
            #------------------------------------#

            # num_features : Number of input features
            # num_classes : Number of output features

            hidden_dim_n = 8
            self.hidden_dim_n = hidden_dim_n

            layers_n = 2
            self.layers_n = layers_n

            #------------------------------------#
            #               Layers               #
            #------------------------------------#

            self.rnn = LSTM(num_features, hidden_dim_n, layers_n, batch_first = True)
            self.fc_output = Sequential(
                 Linear(hidden_dim_n, num_classes),
                 Sigmoid()
            )

        def forward(self, x):

            #------------------------------------#
            #             Pipeline               #
            #------------------------------------#

            batch_size = x.size[0]

            hidden = self.init_hidden(batch_size)

            out, hidden = self.rnn(x, hidden)

            out = out.contiguous().view(-1, self.hidden_dim)

            out = self.fc(out)
        
            return out, hidden
    
        def init_hidden(self, batch_size):
            # Generate first input tensor
            hidden = torch.zeros(self.layers_n, batch_size, self.hidden_dim_n)
            return hidden