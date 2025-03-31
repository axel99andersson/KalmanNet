import torch
import torch.nn as nn

from .KalmanNet_nn import KalmanNetNN
from .RecoveryController import RecoveryController

class RecoveryNetwork(nn.Module):
    def __init__(self, kalman_net: KalmanNetNN, controller: RecoveryController, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.kalman_net = kalman_net
        self.controller = controller

    def forward(self, x):
        x = self.kalman_net(x)
        return self.controller(x)
    
    def init_hidden_KNet(self):
        self.kalman_net.init_hidden_KNet()

    def initSequence(self, ):
        self.kalman_net.InitSequence()
    



