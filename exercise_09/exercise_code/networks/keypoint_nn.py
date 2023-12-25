"""Models for facial keypoint detection"""

import torch
import torch.nn as nn
import pytorch_lightning as pl

class KeypointModel(nn.Module):
# class KeypointModel(pl.LightningModule):
    """Facial keypoint detection model"""
    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
            
        NOTE: You could either choose between pytorch or pytorch lightning, 
            by switching the class name line.
        """
        super().__init__()
        self.save_hyperparameters(hparams)
        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        #                                                                      #
        # We would truly recommend to make your code generic, such as you      #
        # automate the calculation of the number of parameters at each layer.  #
        # You're going probably try different architecutres, and that will     #
        # allow you to be quick and flexible.                                  #
        ########################################################################
        
        input_size = hparams['input_size']
        output_size = hparams['output_size']
        kernel_size = hparams['kernel_size']
        stride = hparams['stride']
        padding = hparams['padding']
        first_depth = hparams['depth']
        hidden_size = hparams['hidden_size']
        cnn_layer = hparams['cnn_layer']
        fc_layer = hparams['fc_layer']
        
        N = input_size
        K = kernel_size
        P = padding
        S = stride
        H = first_depth

        self.model = nn.Sequential()

        # Convolutional layers 
        out = N 
        in_channels = 1
        out_channels = H

        for i in range(cnn_layer):
            self.model.add_module("conv" + str(i+1), nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=P))
            self.model.add_module("max_pool" + str(i+1), nn.MaxPool2d(kernel_size=2, stride=S, padding=P))
            self.model.add_module("relu" + str(i+1), nn.ReLU())

            out = ((out-K+2*P)/S)+1   # after filter 
            out = ((out-2+2*P)/S)+1   # after pooling 
            in_channels = out_channels
            out_channels *= 2

        # Flatten before fully connected layers
        self.model.add_module("flatten", nn.Flatten())

        # Linear layers 
        input_size = out_channels * N * N
        output_size = hidden_size

        for i in range(fc_layer):
            self.model.add_module("fc" + str(i+1), nn.Linear(in_features=input_size, out_features=output_size))
            self.model.add_module("relu_fc" + str(i+1), nn.ReLU())
            input_size = output_size
            output_size = output_size // 2

        # Output layer 
        self.model.add_module("output", nn.Linear(in_features=input_size, out_features=output_size))

        # NOTE: How to calculate the number of model parameters? 
        # self.conv = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2))
        # N_out = (N_in - K + 2P) / S 
        # model_param_num = out_channels * (in_channels* K * K + 1)
        # Here, in_channels is the kernel depth, out_channels is the number of feature maps.
        # 1 is added for the bias parameter of each kernel.

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        
        # check dimensions to use show_keypoint_predictions later
        if x.dim() == 3:
            x = torch.unsqueeze(x, 0)
        ########################################################################
        # TODO: Define the forward pass behavior of your model                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted keypoints.                                   #
        ########################################################################

        x = self.model(x)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return x


class DummyKeypointModel(pl.LightningModule):
    """Dummy model always predicting the keypoints of the first train sample"""
    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)
