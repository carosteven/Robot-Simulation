import torch.nn as nn
import torch.nn.functional as F
from networks import resnet18

class SensorDQN(nn.Module):
    """
    feed-forward nn
    takes difference between current and previous screen patch
    tries to predict the expected return of taking each action given current input
    """
    def __init__(self, n_observations, n_actions):
        super(SensorDQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
    
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...])
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
class VisionDQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(VisionDQN, self).__init__()
        self.resnet18 = resnet18(num_input_channels=1, num_classes = n_actions)
    
    def forward(self, x):
        return self.resnet18(x)