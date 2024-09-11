import matplotlib
import math
import random
import numpy as np
from  collections import namedtuple, deque
from itertools import count
import yaml
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import environments
import models

with open('configurations/config_push_small_sln_my_rews.yml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

env = environments.selector(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'distance'))

# Get number of actions from env
n_actions = len(env.available_actions)
action_freq = 25
# Get numbaer of state observations
state = torch.tensor(env.reset(), dtype=torch.int32, device=device).unsqueeze(0)
n_observations = 3 #len(state)
# checkpoint_path = 'model - no pushing.pt'
checkpoint_path = 'model_weights/model_push_rews.pt'

options = config['options']

if options == True:
    policies = []
    for i in range(config['num_policies']):
        if i == 0:
            policy_net = models.VisionDQN(n_observations, n_actions=2)
        else:
            policy_net = models.VisionDQN_SAM(n_observations)
        
        policy_net.eval()
        checkpoint = torch.load(checkpoint_path[:-3]+f'_{i}.pt', map_location=device)
        policy_net.load_state_dict(checkpoint)
        policies.append(policy_net)

else:
    if config['state_type'] == 'vision':
        if config['model'] == 'resnet':
            if config['action_type'] == 'straight-line-navigation':
                policy_net = models.VisionDQN_SAM(n_observations)
            else:
                policy_net = models.VisionDQN(n_observations, n_actions)
    policy_net.eval()

    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy_net.load_state_dict(checkpoint)

def transform_state(state_batch):
        colour_batch = torch.zeros((state_batch.shape[0], 3, state_batch.shape[2], state_batch.shape[3]),device=device)
        # print(state_batch[:,0])
        colour_batch[:,0] = torch.bitwise_right_shift(state_batch[:,0], 16)
        colour_batch[:,1] = torch.bitwise_right_shift(state_batch[:,0], 8&255)
        colour_batch[:,2] = torch.bitwise_and(state_batch[:,0], 255)
        # np_array = colour_batch.cpu().numpy()
        # np_array = np_array.transpose(0,2,3,1)
        # image = Image.fromarray(np_array[0].astype(np.uint8))
        # image.show()
        # input()
        return colour_batch

def policy(state, policy_n=None):
    if policy_n is None:
        policy_n = policy_net

    with torch.no_grad():
        state = state.to(torch.int32)
        qvalues = policy_n(transform_state(state))
        # print(torch.max(qvalues).item())
        # print(qvalues[:,152*76+76])
        action = torch.argmax(qvalues).item()

    action = torch.tensor([[action]], device=device, dtype=torch.long)
    
    return action

def sln_action_control(state, policy_n=None):
    action = policy(state, policy_n)
    action = np.unravel_index(action[0,0].cpu(), (int(env.screen_size[0]/2), int(env.screen_size[1]/2)))
    action = (action[0]*2, action[1]*2)

    total_reward = 0
    while not env.action_completed:
        observation, reward, done, _ = env.step(action, test=True)
        total_reward += reward
    print(total_reward)
    env.action_completed = False

    if done:
        next_state = None
    else:
        next_state = torch.tensor(observation, dtype=torch.int32, device=device).unsqueeze(0)
        
    total_reward = torch.tensor([total_reward], device=device)

    state = next_state
    return done, state

def primitive_action_control(state, action):
    print("Back it up")
    total_reward = 0
    for frame in range(action_freq):
        if frame == 0:
            _, reward, done, _ = env.step(env.available_actions[action], primitive=True)

        elif frame == action_freq - 1:
            env.action_completed = True
            next_state, reward, done, _ = env.step(None, primitive=True)
            if done:
                next_state = None
            else:
                next_state = torch.tensor(next_state, dtype=torch.int32, device=device).unsqueeze(0)
                
            # reward = torch.tensor([reward], device=device)

        else:
            _, reward, done, _ = env.step(None, primitive=True)
        
        total_reward += reward
        if done:
            next_state = None
            break
    
    env.action_completed = False
    print(total_reward)
    total_reward = torch.tensor([total_reward], device=device)
    state = next_state
    return done, state

state = torch.tensor(env.reset(), dtype=torch.int32, device=device).unsqueeze(0)
done = False
# print(checkpoint['epoch'])
frame = 0
while not done:
    if options == True:
        option = policy(state, policies[0])
        if option == 0:
            done, state = primitive_action_control(state, action=1)
        elif option == 1:
            done, state = sln_action_control(state, policies[1])
        continue

    '''
    if frame % action_freq == 0:
        action = policy(transform_state(state))
        state, reward, done, info = env.step(env.available_actions[action])
        state = torch.tensor(state, dtype=torch.int32, device=device).unsqueeze(0)
        print(action, end='\r')
        # print(f'{round(env.reward, 2)} ', end='\r')
    '''
    if config['action_type'] == 'straight-line-navigation':
        done, state = sln_action_control(state)
    else:
        env.step(None)
        
    if done:
        state = torch.tensor(env.reset(), dtype=torch.int32, device=device).unsqueeze(0)
        done = False
    frame += 1