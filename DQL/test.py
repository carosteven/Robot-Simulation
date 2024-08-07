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
action_freq = 10
# Get numbaer of state observations
state = env.reset()
n_observations = len(state)
# checkpoint_path = 'model - no pushing.pt'
checkpoint_path = 'model_push.pt'


policy_net = models.VisionDQN(n_observations, n_actions).to(device)
policy_net.eval()

checkpoint = torch.load(checkpoint_path, map_location=device)
policy_net.load_state_dict(checkpoint)

def select_action(state):
    # return policy_net(state).max(1).indices.view(1,1)
    qvalues = policy_net(state)
    action = torch.argmax(qvalues).item()
    return torch.tensor([[action]], device=device, dtype=torch.long)

state = env.reset()
state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
done = False
# print(checkpoint['epoch'])
frame = 0
while not done:
    if frame % action_freq == 0:
        action = select_action(state)
        state, reward, done, info = env.step(env.available_actions[action])
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        print(action, end='\r')
        # print(f'{round(env.reward, 2)} ', end='\r')
    else:
        env.step(None)
        
    if done:
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
    frame += 1