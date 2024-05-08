
import torch
from torch.nn import Module, Sequential
from torch.nn import Linear, Conv2d, AdaptiveAvgPool2d, ReLU, Sigmoid, Flatten

#---------------------------------------------------------------------------------------------------#
#                                          Conv Block                                               #
#---------------------------------------------------------------------------------------------------#

class Conv_Block(Module):
        def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
            super(Conv_Block, self).__init__()

            #------------------------------------#
            #          Hyperparameters           #
            #------------------------------------#

            # in_channels : Number of input channels
            # out_channels : Number of output channels
            # kernel_size : Kernel size
            # stride : Stride
            # padding : Padding

            activation_function = ReLU()

            #------------------------------------#
            #               Layers               #
            #------------------------------------#

            self.conv = Sequential(
                Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                activation_function
            )     

        def forward(self, x):

            #------------------------------------#
            #             Pipeline               #
            #------------------------------------#

            output = self.conv(x)

            return output
        
#---------------------------------------------------------------------------------------------------#
#                                Convolutional Neural Network                                       #
#---------------------------------------------------------------------------------------------------#

class Conv_Neural_Network(Module):
        def __init__(self, num_features: int, num_classes: int):
            super(Conv_Neural_Network, self).__init__()

            #------------------------------------#
            #          Hyperparameters           #
            #------------------------------------#

            # num_features : Number of input features
            # num_classes : Number of output features

            layer_1_filters = 8
            layer_2_filters = 16
            layer_3_filters = 32

            #------------------------------------#
            #               Layers               #
            #------------------------------------#

            self.conv_input = Conv_Block(num_features, layer_1_filters, 3, 2, 1)
            self.conv_l1 = Conv_Block(layer_1_filters, layer_2_filters, 3, 2, 1)
            self.conv_l2 = Conv_Block(layer_2_filters, layer_3_filters, 3, 2, 1)



            #----------------------------------------------#

            self.conv_output = Sequential(
                 AdaptiveAvgPool2d(1),
                 Flatten(),
                 Linear(layer_3_filters, num_classes),
                 Sigmoid()
            )

            #----------------------------------------------#

        def forward(self, x):

            #------------------------------------#
            #             Pipeline               #
            #------------------------------------#

            x = self.conv_input(x)
            x = self.conv_l1(x)
            x = self.conv_l2(x)
            output = self.conv_output(x)

            return output
        
#---------------------------------------------------------------------------------------------------#
#                                Fully Convolutional Neural Network                                 #
#---------------------------------------------------------------------------------------------------#

class Fully_Conv_Neural_Network(Module):
        def __init__(self, num_features: int, num_classes: int):
            super(Fully_Conv_Neural_Network, self).__init__()

            #------------------------------------#
            #          Hyperparameters           #
            #------------------------------------#

            # num_features : Number of input features
            # num_classes : Number of output features

            layer_1_filters = 8
            layer_2_filters = 16
            layer_3_filters = 32

            #------------------------------------#
            #               Layers               #
            #------------------------------------#

            self.conv_input = Conv_Block(num_features, layer_1_filters, 3, 2, 1)
            self.conv_l1 = Conv_Block(layer_1_filters, layer_2_filters, 3, 2, 1)
            self.conv_l2 = Conv_Block(layer_2_filters, layer_3_filters, 3, 2, 1)



            #----------------------------------------------#

            self.conv_output = Sequential(
                 AdaptiveAvgPool2d(1),
                 Conv_Block(layer_3_filters, num_classes, 1, 0, 0),
                 Sigmoid()
            )

            #----------------------------------------------#

        def forward(self, x):

            #------------------------------------#
            #             Pipeline               #
            #------------------------------------#

            x = self.conv_input(x)
            x = self.conv_l1(x)
            x = self.conv_l2(x)
            output = self.conv_output(x)

            return output
    