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
        
        # Define the convolutional layers
        '''
        ConvLayer output size = (Input size - Kernel size + 2 * Padding) / Stride + 1
        '''
        # (224-3+2)/2 + 1 = 224 => After pool = 112
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=1, bias=False)
        self.batchNorm1 = nn.BatchNorm2d(num_features=8,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        # (112-3+2)/2 + 1 = 112 => After pool = 56
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1, bias=False)
        self.batchNorm2 = nn.BatchNorm2d(num_features=16,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        
        # (56-3+2)/2 + 1 = 526 => After pool = 28
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(num_features=32,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # (28-3+2)/2 + 1 = 28 => After pool = 14
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(num_features=32,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        
        # (28-3+2)/2 + 1 = 28 => After pool = 7
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.batchNorm5 = nn.BatchNorm2d(num_features=64,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        
        '''
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.batchNorm6 = nn.BatchNorm2d(num_features=64,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.batchNorm7 = nn.BatchNorm2d(num_features=128,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
                
        self.conv8 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.batchNorm8 = nn.BatchNorm2d(num_features=128,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        '''        

        self.fc1 = nn.Linear(in_features=64*7*7, out_features=1024, bias=True)
        self.batchNorm9 = nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fc2 = nn.Linear(in_features=1024, out_features=136, bias=True)
        
        
   
    def forward(self, x):
        # Apply the convolutional and pooling layers
        x = F.relu(self.batchNorm1(self.conv1(x)))
        x = F.relu(self.batchNorm2(self.conv2(x)))
        x = F.relu(self.batchNorm3(self.conv3(x)))
        x = F.relu(self.batchNorm4(self.conv4(x)))
        x = F.relu(self.batchNorm5(self.conv5(x)))
        #x = F.relu(self.batchNorm6(self.conv6(x)))
        #x = F.relu(self.batchNorm7(self.conv7(x)))
        #x = F.relu(self.batchNorm8(self.conv8(x)))
        
        # Apply the convolutional layer to replace the flatten layer
        # x = self.conv_flatten(x)
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        #x = x.view(-1, 1024)

        # Apply the fully connected layers
        x = F.relu(self.fc1(x))
        #x = self.batchNorm9(x)
        x = self.fc2(x)
        return x

