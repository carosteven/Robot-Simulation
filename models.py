import torch.nn as nn
import torch.nn.functional as F
from networks import resnet18, densenet121

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
        self.resnet18 = resnet18(num_input_channels=n_observations, num_classes = n_actions)
    
    def forward(self, x):
        return self.resnet18(x)
    
class VisionDQN_SAM(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(VisionDQN_SAM, self).__init__()
        self.resnet18 = resnet18(num_input_channels=n_observations, num_classes = n_actions)
        self.conv1 = nn.Conv2d(512, 128, kernel_size=1, padding=1)
        self.conv2 = nn.Conv2d(128, 32, kernel_size=1, padding=1)
        self.conv3 = nn.Conv2d(32, n_actions, kernel_size=1, padding=1)
    
    def forward(self, x):
        x = self.resnet18.features(x)
        x = F.relu(self.conv1(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.conv2(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return self.conv3(x)

    
class VisionDQN_dense(nn.Module):
    def __init__(self, n_channels, n_actions):
        super(VisionDQN_dense, self).__init__()
        self.densenet121 = densenet121(num_input_channels=n_channels, num_classes = n_actions, growth_rate=8)

    def forward(self, x):
        return self.densenet121(x)