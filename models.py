import torch
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
    def __init__(self, n_observations):
        super(VisionDQN_SAM, self).__init__()
        self.resnet18 = resnet18(num_input_channels=n_observations)
        self.conv1 = nn.Conv2d(512, 128, kernel_size=1)#, padding=1)
        self.conv2 = nn.Conv2d(128, 32, kernel_size=1)#, padding=1)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=1)#, padding=1)
    
    def forward(self, x):
        x = self.resnet18.features(x)
        x = F.relu(self.conv1(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.conv2(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return torch.flatten(self.conv3(x), 1)

    
class VisionDQN_dense(nn.Module):
    def __init__(self, n_channels, n_actions):
        super(VisionDQN_dense, self).__init__()
        self.densenet121 = densenet121(num_input_channels=n_channels, num_classes = n_actions, growth_rate=8)

    def forward(self, x):
        return self.densenet121(x)
    
class VisionDQN_Multi_Cam(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(VisionDQN_Multi_Cam, self).__init__()
        # Overhead Camera
        self.convOC1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.convOC2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.convOC3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.maxpoolOC = nn.MaxPool2d(2, 2)
        self.fcOC1 = nn.Linear(64*17*17, 512)

        # Front Camera
        self.fcFCr = nn.Linear(3*3, 16)
        self.fcFCg = nn.Linear(3*3, 16)
        self.fcFCb = nn.Linear(3*3, 16)
        self.fcFC = nn.Linear(16*3, 32)

        # Attention
        self.attention_fc = nn.Linear(512+32, 2)
        
        # Combined Layers
        self.fc1 = nn.Linear(512+32, 256)
        self.fc2 = nn.Linear(256, n_actions)

        

    def forward(self, OC_FC):
        OC, FC = OC_FC
        OC = F.relu(self.convOC1(OC))
        OC = F.relu(self.convOC2(OC))
        OC = F.relu(self.convOC3(OC))
        OC = self.maxpoolOC(OC)
        OC = torch.flatten(OC, 1)
        OC = F.relu(self.fcOC1(OC))

        FCr = FC[:, 0].view(FC[:, 0].size(0), -1)
        FCr = F.relu(self.fcFCr(FCr))

        FCg = FC[:, 1].view(FC[:, 1].size(0), -1)
        FCg = F.relu(self.fcFCg(FCg))
        
        FCb = FC[:, 2].view(FC[:, 2].size(0), -1)
        FCb = F.relu(self.fcFCb(FCb))

        FC = torch.cat((FCr, FCg, FCb), 1)
        FC = F.relu(self.fcFC(FC))

        # Attention
        OC_FC = torch.cat((OC, FC), 1)
        attention_weights = F.softmax(self.attention_fc(OC_FC), dim=1)
        OC = OC * attention_weights[:, 0].unsqueeze(1)
        FC = FC * attention_weights[:, 1].unsqueeze(1)

        x = torch.cat((OC, FC), 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)