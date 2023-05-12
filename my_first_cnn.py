"""
Code implementing my own CNN. Start from a very basic understanding, and then build 
and experiment to see what's gonna succeed in performing well.

I might use the article below as learning source for implementing "LeNet".
https://pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyFirstCNN(nn.Module):

    def __init__(self, 
                 input_shape : tuple[int, int, int],
                 output_size : int
                 ):
        """
        Creates a CNN with convolution - max pool stacks.

        :param Tuple input_size: A tuple (channel, height, width) for input shape of image.
        :param int output_size: The size of output equal to the number of classes we predict for.
        """
        super(MyFirstCNN, self).__init__()

        num_channel = input_shape[0]
        l1_num_kernel = 16
        l2_num_kernel = l1_num_kernel * 2
        
        self.conv_mpool_stack = nn.Sequential(
            nn.Conv2d( #24x24x3 -> 22x22x16
                in_channels=num_channel,
                out_channels=l1_num_kernel,
                kernel_size=3
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), #22x22x16 -> 11x11x16
            nn.Conv2d(l1_num_kernel, l2_num_kernel, 3), #11x11x16 -> 10x10x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) #10x10x32 -> 5x5x32
        )
        self.fc1 = nn.Linear( #800 -> 1600
            in_features= 5 * 5 * l2_num_kernel, 
            out_features= 5 * 5 * l2_num_kernel * 2
        )
        self.fc2 = nn.Linear(5*5*l2_num_kernel*2, output_size) #1600 -> 10
    
    def forward(self, x : torch.tensor):
        """
        Evaluates the given tensor in this model.

        :param torch.tensor x: The given tensor of shape (batch, channel, height, width).
        :return torch.tensor: The evaluated tensor to be returned.
        """
        x = self.conv_mpool_stack(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x