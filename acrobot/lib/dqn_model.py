import torch
import torch.nn as nn
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

#        self.conv = nn.Sequential(
#                nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
#                nn.ReLU(),
#                nn.Conv2d(32, 64, kernel_size=4, stride=2),
#                nn.ReLU(),
#                nn.Conv2d(64, 64, kernel_size=3, stride=1),
#                nn.ReLU()
#        )
#        
        self.net = nn.Sequential(
                nn.Linear(input_shape[0], 512),
                nn.ReLU(),
                nn.Linear(512, n_actions)
            )

#        conv_out_size = self._get_conv_out(input_shape)
#        self.fc = nn.Sequential(
#                nn.Linear(conv_out_size, 512),
#                nn.ReLU(),
#                nn.Linear(512, n_actions)
#        )
#
#    def _get_conv_out(self, shape):
#        print('shape', shape)
#        o = self.conv(torch.zeros(1, *shape))
#        return int(np.prod(o.size()))
#
#    def forward(self, x):
#        conv_out = self.conv(x).view(x.size()[0], -1)
#        print('conv', conv_out)
#        return self.fc(conv_out)
#
    def forward(self, x):
        x = x.float()
        return self.net(x)
