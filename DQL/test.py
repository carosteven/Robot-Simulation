import matplotlib
import math
import random
import numpy as np
from  collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import sys
sys.path.insert(1, './environments')
from wheeled_robot_env import Wheeled_Robot_Sim

env = Wheeled_Robot_Sim(state_type='')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Get number of actions from env
n_actions = len(env.available_actions)
# Get numbaer of state observations
state = env.reset()
n_observations = len(state)
checkpoint_path = 'model - no pushing.pt'

class DQN(nn.Module):
    """
    feed-forward nn
    takes difference between current and previous screen patch
    tries to predict the expected return of taking each action given current input
    """
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
    
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...])
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

policy_net = DQN(n_observations, n_actions).to(device)
policy_net.eval()

checkpoint = torch.load(checkpoint_path)
policy_net.load_state_dict(checkpoint['policy_state_dict'])

def select_action(state):
    return policy_net(state).max(1).indices.view(1,1)

state = env.reset()
state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
done = False
while not done:
    action = select_action(state)
    state, reward, done, info = env.step(env.available_actions[action])
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    if done:
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
print(reward)