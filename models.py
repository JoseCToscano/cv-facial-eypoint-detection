## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## Definition all the layers of this CNN:
        
        ## 1. This network takes in a square (same width and height), grayscale image as input
        # 3x3 square convolution kernel
        ## output size = (W-F)/S +1 = (224-3)/1 +1 = 222
        # the output Tensor for one image, will have the dimensions: (10, 222, 222)
        # after one pool layer, this becomes (10, 111, 111)
        self.conv1 = nn.Conv2d(1, 32, 3)
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)
        
        # second conv layer: 10 inputs, 20 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (111-3)/1 +1 = 109
        # the output tensor will have dimensions: (20, 54, 54)
        # after another pool layer this becomes (20, 54, 54); 5.5 is rounded down
        self.conv2 = nn.Conv2d(32,64, 3)
        
        # 20 outputs * the 54*54 filtered/pooled map size
        self.fc1 = nn.Linear(64*54*54, 4096)
        
        # dropout with p=0.4
        self.fc1_drop = nn.Dropout(p=0.4)
        
        # finally, create 10 output channels (for the 10 classes)
        # finally, create 136 output channels, 2 for each of the 68 keypoint (x, y) pairs        
        self.fc2 = nn.Linear(4096, 136)


        
    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        # two linear layers with dropout in between
        x = F.relu(self.fc1(x)) 
        x = self.fc1_drop(x)
        x = self.fc2(x)

        # Weight initializarion using Xavier's method
        # self.initialize_weights()
        # a modified x, having gone through all the layers of your model, should be returned
        return x
