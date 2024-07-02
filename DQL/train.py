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
logging.getLogger('pymunk').propagate = False

from nav_obstacle_env import Nav_Obstacle_Env
from push_empty_env import Push_Empty_Env
import models

# env = Nav_Obstacle_Env()
env = None
'''
TODO add more info to log
TODO make resnet predict two actions
TODO use multiple GPUs
'''

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



class Train_DQL():
    def __init__(self, state_type, checkpoint_path, checkpoint_interval, num_epoch):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.checkpoint_path = checkpoint_path
        self.checkpoint_interval = checkpoint_interval
        self.num_epoch = num_epoch
        # Get number of actions from env
        self.n_actions = len(env.available_actions)
        # Get number of state observations
        self.state = env.reset() # state, info = env.reset()
        self.n_observations = len(self.state)
        
        self.create_or_restore_training_state(state_type)

        self.steps_done = 0 # for exploration
        self.first_contact_made = False # end episode if agent does not push box after x actions

    def create_or_restore_training_state(self, state_type):
        if state_type == 'vision':
            self.policy_net = models.VisionDQN(self.n_observations, self.n_actions)
            self.target_net = models.VisionDQN(self.n_observations, self.n_actions)
            
        else:    
            self.policy_net = models.SensorDQN(self.n_observations, self.n_actions)
            self.target_net = models.SensorDQN(self.n_observations, self.n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(10000)
        self.epoch = 0
        self.loss = 0

        if os.path.exists(self.checkpoint_path):
            training_state = torch.load(self.checkpoint_path)
            self.policy_net.load_state_dict(training_state['policy_state_dict'])
            self.target_net.load_state_dict(training_state['target_state_dict'])
            self.optimizer.load_state_dict(training_state['optimizer_state_dict'])
            self.memory.memory = training_state['memory']
            self.epoch = training_state['epoch']
            self.loss = training_state['loss']
            logging.info(f"Training state restored at epoch {self.epoch}")
        else:
            logging.info("No checkpoint detected, starting from initial state")

    def commit_state(self):
        temp_path = os.path.join(os.path.dirname(self.checkpoint_path), "temp.pt")
        print(self.epoch)
        training_state = {
            'policy_state_dict' : self.policy_net.state_dict(),
            'target_state_dict' : self.target_net.state_dict(),
            'optimizer_state_dict' : self.optimizer.state_dict(),
            'memory' : self.memory.memory,
            'epoch' : self.epoch,
            'loss' : self.loss, 
        }

        # first save the temp file
        torch.save(training_state, temp_path)
        # according to the GNU spec of rename, the state of checkpoint_path
        # is atomic, i.e. it will either be modified or not modified, but not in
        # between, during a system crash (i.e. preemtion)
        os.replace(temp_path, self.checkpoint_path)
        msg = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ": Checkpoint saved at " + self.checkpoint_path
        logging.info(msg)

    def select_action(self):
        sample = random.random()
        eps_treshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1

        if sample > eps_treshold:
            with torch.no_grad():
                # t.max(1) will return the largect column value of each row
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expeced reward.
                # print(policy_net(state).shape)
                return self.policy_net(self.state).max(1).indices.view(1,1)
        else:
            return torch.tensor([[random.randint(0, self.n_actions-1)]], device=self.device, dtype=torch.long)

    episode_durations = []

    def plot_durations(self, show_results=False):
        pass

    def optimize_model(self):
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        self.loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        self.loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def optimizer_to_dev(self):
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)

    def train(self):
        start_time = time.time()
        self.policy_net = self.policy_net.to(self.device)
        self.target_net = self.target_net.to(self.device)
        self.optimizer_to_dev()

        # while self.epoch < self.num_epoch:
        for self.epoch in tqdm(range(self.num_epoch)):
            # print(self.epoch)
            # Initialize the environment and get its state
            self.state = env.reset()
            self.state = torch.tensor(self.state, dtype=torch.float32, device=self.device).unsqueeze(0)
            self.first_contact_made = False
            for t in count():
            # for t in tqdm(range(100000)):
                # print(t, end='\r')
                action = self.select_action()
                observation, reward, done, info = env.step(env.available_actions[action])
                reward = torch.tensor([reward], device=self.device)

                if env.is_pushing:
                    self.first_contact_made = True

                if t > 5000 and not self.first_contact_made:
                    done = True
                    logging.info("No contact made. Resetting environment...")

                if done:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                
                # Store the transition in memory
                self.memory.push(self.state, action, next_state, reward)

                # Move to the next state
                self.state = next_state

                # Perform one step of the optimization (on the policy network)
                if len(self.memory) >= BATCH_SIZE*5:
                    self.optimize_model()

                # Soft update of the target network's weights
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                self.target_net.load_state_dict(target_net_state_dict)

                cur_time = time.time()
                if cur_time - start_time > self.checkpoint_interval:
                    self.commit_state()
                    start_time = cur_time

                if done:
                    self.episode_durations.append(t+1)
                    self.plot_durations()
                    break
            # print()
            # self.epoch += 1

def env_selector(env_num):
    if env_num == 0:
        return Nav_Obstacle_Env()
    elif env_num == 1:
        return Push_Empty_Env()
    else:
        print("Bad environment selection")
        return None

def main(args):
    global BATCH_SIZE, env
    if args.log_file is not None:
        logging.basicConfig(filename=args.log_file,level=logging.DEBUG)
    
    logging.info("starting training script")

    env = env_selector(args.environment)
    env.state_type = args.state_type
    
    BATCH_SIZE = args.batch_size

    train = Train_DQL(args.state_type, args.checkpoint_path, args.checkpoint_interval, args.num_epoch)
    
    # check if the checkpoint exists and try to resume from the last checkpoint
    # if you are saving for every epoch, you can skip the part about
    # saving and loading the dataloader state.
    
    # policy_net, target_net, optimizer, memory, epoch, loss = \
    #     create_or_restore_training_state(args.state_type, args.batch_size, checkpoint_path)
    
    train.train()


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

    parser.add_argument(
        '--environment',
        type=int,
        help='environment to simulate- 0: nav_obstacle, 1: push_empty',
        default=0
    )

    main(parser.parse_args())