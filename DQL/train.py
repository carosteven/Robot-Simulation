import matplotlib
import math
import random
import numpy as np
import argparse
import time
from  collections import namedtuple, deque
from itertools import count
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

import os, sys
sys.path.insert(1, './environments')

import logging

from wheeled_robot_env import Wheeled_Robot_Sim
import models

env = Wheeled_Robot_Sim()

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

BATCH_SIZE = 128    # BATCH_SIZE is the number of transitions sampled from the replay buffer
GAMMA = 0.99        # GAMMA is the discount factor
EPS_START = 0.9     # EPS_START is the starting value of epsilon
EPS_END = 0.05      # EPS_END is the final value of epsilon
EPS_DECAY = 1000    # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
TAU = 0.005         # TAU is the update rate of the target network
LR = 1e-4           # LR is the learning rate of the ``AdamW`` optimizer

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Get number of actions from env
n_actions = len(env.available_actions)
# Get number of state observations
# state, info = env.reset()
state = env.reset()
n_observations = len(state)

def create_or_restore_training_state(state_type, batch_size, checkpoint_path):
    if state_type == 'vision':
        policy_net = models.VisionDQN(n_observations, n_actions)
        target_net = models.VisionDQN(n_observations, n_actions)
        
    else:    
        policy_net = models.SensorDQN(n_observations, n_actions)
        target_net = models.SensorDQN(n_observations, n_actions)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)
    epoch = 0
    loss = 0

    if os.path.exists(checkpoint_path):
        training_state = torch.load(checkpoint_path)
        policy_net.load_state_dict(training_state['policy_state_dict'])
        target_net.load_state_dict(training_state['target_state_dict'])
        optimizer.load_state_dict(training_state['optimizer_state_dict'])
        memory.memory = training_state['memory']
        epoch = training_state['epoch']
        loss = training_state['loss']
        logging.info(f"Training state restored at epoch {epoch}")
    else:
        logging.info("No checkpoint detected, starting from initial state")
    
    return policy_net, target_net, optimizer, memory, epoch, loss

def commit_state(policy_net, target_net, optimizer, memory, epoch, loss, checkpoint_path):
    temp_path = os.path.join(os.path.dirname(checkpoint_path), "temp.pt")

    training_state = {
        'policy_state_dict' : policy_net.state_dict(),
        'target_state_dict' : target_net.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'memory' : memory.memory,
        'epoch' : epoch,
        'loss' : loss, 
    }

    # first save the temp file
    torch.save(training_state, temp_path)
    # according to the GNU spec of rename, the state of checkpoint_path
    # is atomic, i.e. it will either be modified or not modified, but not in
    # between, during a system crash (i.e. preemtion)
    os.replace(temp_path, checkpoint_path)
    msg = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ": Checkpoint saved at " + checkpoint_path
    logging.info(msg)

steps_done = 0

def select_action(state, policy_net, device):
    global steps_done
    sample = random.random()
    eps_treshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    if sample > eps_treshold:
        with torch.no_grad():
            # t.max(1) will return the largect column value of each row
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expeced reward.
            # print(policy_net(state).shape)
            return policy_net(state).max(1).indices.view(1,1)
    else:
        return torch.tensor([[random.randint(0, n_actions-1)]], device=device, dtype=torch.long)

episode_durations = []

def plot_durations(show_results=False):
    pass

def optimize_model(policy_net, target_net, optimizer, memory, loss, device):
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    return policy_net, optimizer, memory, loss

num_epochs = 50
num_samples = 1000

def optimizer_to_dev(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

def train(policy_net, target_net, optimizer, memory, epoch, loss, checkpoint_path, checkpoint_interval, max_epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_time = time.time()
    policy_net = policy_net.to(device)
    target_net = target_net.to(device)
    optimizer_to_dev(optimizer, device)

    for i_epochs in tqdm(range(max_epoch)):
        # Initialize the environment and get its state
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        # print(state.shape)
        # for t in count():
        for t in tqdm(range(num_samples)):
            action = select_action(state, policy_net, device)
            observation, reward, done, info = env.step(env.available_actions[action])
            reward = torch.tensor([reward], device=device)

            if done:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            
            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            if len(memory) >= BATCH_SIZE:
                policy_net, optimizer, memory, loss = \
                    optimize_model(policy_net, target_net, optimizer, memory, loss, device)

            # Soft update of the target network's weights
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            cur_time = time.time()
            if cur_time - start_time > checkpoint_interval:
                commit_state(policy_net, target_net, optimizer, memory, epoch, loss, checkpoint_path)
                start_time = cur_time

            if done:
                episode_durations.append(t+1)
                plot_durations()
                break

def main(args):
    global BATCH_SIZE
    if args.log_file is not None:
        logging.basicConfig(filename=args.log_file,level=logging.DEBUG)
    
    logging.info("starting training script")
    checkpoint_path = args.checkpoint_path

    env.state_type = args.state_type
    BATCH_SIZE = args.batch_size
    
    # check if the checkpoint exists and try to resume from the last checkpoint
    # if you are saving for every epoch, you can skip the part about
    # saving and loading the dataloader state.
    policy_net, target_net, optimizer, memory, epoch, loss = \
        create_or_restore_training_state(args.state_type, args.batch_size, checkpoint_path)
    
    policy_net = train(policy_net, target_net, optimizer, memory, epoch, loss, checkpoint_path, args.checkpoint_interval, args.num_epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="trains robot to navigate an environment using DRL"
    )

    parser.add_argument(
        '--state_type',
        type=str,
        help='options: ["vision", "sensor"]',
        required=True
    )

    parser.add_argument(
        '--num_epoch',
        type=int,
        help='number of epochs to run',
        required=True
    )

    parser.add_argument(
        '--checkpoint_path',
        type=str,
        help='path to save and look for the checkpoint file',
        default=os.path.join(os.getcwd(), "checkpoint.pt")
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        help='batch size per iteration',
        default=128
    )

    parser.add_argument(
        '--checkpoint_interval',
        type=int,
        help='period to take checkpoints in seconds',
        default=3600
    )

    parser.add_argument(
        '--log_file',
        type=str,
        help='specify the loaction of the output directory, default stdout',
        default=None
    )

    main(parser.parse_args())