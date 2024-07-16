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

import environments
import models

env = environments.selector(2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Get number of actions from env
n_actions = len(env.available_actions)
# Get numbaer of state observations
state = env.reset()
n_observations = len(state)
# checkpoint_path = 'model - no pushing.pt'
action_path = 'actions.pt'


# policy_net = models.VisionDQN(n_observations, n_actions).to(device)
# policy_net.eval()

# checkpoint = torch.load(action_path, map_location=device)
actions = torch.load(action_path, map_location=device)
print(actions[0])
input
# policy_net.load_state_dict(checkpoint)

# def select_action(state):
    # return policy_net(state).max(1).indices.view(1,1)

state = env.reset()
state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
done = False
# print(checkpoint['epoch'])
while not done:
    action = actions[0]
    actions.pop(0)
    state, reward, done, info = env.step(env.available_actions[action])
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    print(action, end='\r')
    if done:
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
print(reward)