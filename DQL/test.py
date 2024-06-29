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
from nav_obstacle_env import Nav_Obstacle_Env
from push_empty_env import Push_Empty_Env
import models

# env = Wheeled_Robot_Sim(state_type='')
env = Push_Empty_Env()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Get number of actions from env
n_actions = len(env.available_actions)
# Get numbaer of state observations
state = env.reset()
n_observations = len(state)
# checkpoint_path = 'model - no pushing.pt'
checkpoint_path = 'checkpoint/checkpoint.pt'


policy_net = models.VisionDQN(n_observations, n_actions).to(device)
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