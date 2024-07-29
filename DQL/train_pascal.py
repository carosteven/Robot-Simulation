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

import os

import logging
logging.getLogger('pymunk').propagate = False

import environments
import models

env = None

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))



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
    def __init__(self, state_type, model, checkpoint_path, checkpoint_interval, num_epoch, batch_size=128):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.checkpoint_interval = checkpoint_interval
        self.num_epoch = num_epoch
        # Get number of actions from env
        self.n_actions = len(env.available_actions)

        self.action_freq = 25
        # self.action_freq = 1


        # Global variables
        self.BATCH_SIZE = batch_size     # How many examples to sample per train step
        self.GAMMA = 0.99            # Discount factor in episodic reward objective
        self.LEARNING_RATE = 5e-4    # Learning rate for Adam optimizer
        self.TARGET_UPDATE_FREQ = 20   # Target network update frequency
        self.STARTING_EPSILON = 1.0  # Starting epsilon
        self.STEPS_MAX = 200000       # Gradually reduce epsilon over these many steps
        self.EPSILON_END = 0.01      # At the end, keep epsilon at this value

        self.EPSILON = self.STARTING_EPSILON

        # Get number of state observations
        self.state = env.reset() # state, info = env.reset()
        self.n_observations = len(self.state)
        self.next_state = None
        self.action = None
        
        self.create_or_restore_training_state(state_type, model)

        self.steps_done = 0 # for exploration
        self.contact_made = False # end episode if agent does not push box after x actions
        self.last_epi_contact_made = 0

    def create_or_restore_training_state(self, state_type, model):
        if state_type == 'vision':
            if model == 'resnet':
                self.policy_net = models.VisionDQN(self.n_observations, self.n_actions)
                self.target_net = models.VisionDQN(self.n_observations, self.n_actions)
            elif model == 'densenet':
                self.policy_net = models.VisionDQN_dense(self.n_observations, self.n_actions)
                self.target_net = models.VisionDQN_dense(self.n_observations, self.n_actions)
            
        else:    
            self.policy_net = models.SensorDQN(self.n_observations, self.n_actions)
            self.target_net = models.SensorDQN(self.n_observations, self.n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.LEARNING_RATE)
        # self.optimizer = optim.SGD(self.policy_net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LEARNING_RATE, weight_decay=0.01)
        self.memory = ReplayMemory(50000)
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
            self.EPSILON = training_state['epsilon']
            logging.info(f"Training state restored at epoch {self.epoch}")
        else:
            logging.info("No checkpoint detected, starting from initial state")

    def commit_state(self):
        temp_path = os.path.join(os.path.dirname(self.checkpoint_path), "temp.pt")
        training_state = {
            'policy_state_dict' : self.policy_net.state_dict(),
            'target_state_dict' : self.target_net.state_dict(),
            'optimizer_state_dict' : self.optimizer.state_dict(),
            'memory' : self.memory.memory,
            'epoch' : self.epoch,
            'loss' : self.loss,
            'epsilon' : self.EPSILON
        }

        # first save the temp file
        torch.save(training_state, temp_path)
        # according to the GNU spec of rename, the state of checkpoint_path
        # is atomic, i.e. it will either be modified or not modified, but not in
        # between, during a system crash (i.e. preemtion)
        os.replace(temp_path, self.checkpoint_path)
        msg = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ": Checkpoint saved at " + self.checkpoint_path
        logging.info(msg)

        # Update a target network using a source network
    def update_target(self):
        for tp, p in zip(self.target_net.parameters(), self.policy_net.parameters()):
            tp.data.copy_(p.data)
    
    def policy(self):
        # With probability EPSILON, choose a random action
        # Rest of the time, choose argmax_a Q(s, a) 
        if np.random.rand() < self.EPSILON:
            action = np.random.randint(self.n_actions)
        else:
            qvalues = self.policy_net(self.state)
            action = torch.argmax(qvalues).item()
        action = torch.tensor([[action]], device=self.device, dtype=torch.long)
        
        # Epsilon update rule: Keep reducing a small amount over
        # STEPS_MAX number of steps, and at the end, fix to EPSILON_END
        prev_eps = self.EPSILON
        self.EPSILON = max(self.EPSILON_END, self.EPSILON - (1.0 / self.STEPS_MAX))
        if self.EPSILON == self.EPSILON_END and self.EPSILON != prev_eps:
            logging.info("Reached min epsilon")
        # print(EPSILON)

        return action
    
    def update_networks(self, epi):
    
        # Sample a minibatch (s, a, r, s', d)
        # Each variable is a vector of corresponding values
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        # Get Q(s, a) for every (s, a) in the minibatch
        qvalues = self.policy_net(state_batch).gather(1, action_batch.view(-1, 1)).squeeze()
        '''
        # Get max_a' Qt(s', a') for every (s') in the minibatch
        q2values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            q2values[non_final_mask] = torch.max(self.target_net(non_final_next_states), dim = 1).values
        
        # q2value is zero when the state is final ^^
        targets = reward_batch + self.GAMMA * q2values
        '''
        # Double DQN Formula: r + gamma*TARGET(s_t+1, argmax_a POLICY(s_t+1, a))
        q_target_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            actions = torch.argmax(self.policy_net(non_final_next_states), dim=1).values
            q_target_values[non_final_mask] = self.target_net(non_final_next_states)[actions].values
        targets = reward_batch + self.GAMMA * q_target_values
        
        '''
        # If done, 
        #   y = r(s, a) + GAMMA * max_a' Q(s', a') * (0)
        # If not done,
        #   y = r(s, a) + GAMMA * max_a' Q(s', a') * (1)       
        targets = reward_batch + self.GAMMA * q2values * (1-env._done)
        '''
        # Detach y since it is the target. Target values should
        # be kept fixed.
        loss = torch.nn.SmoothL1Loss()(targets.detach(), qvalues)
        # print(qvalues[0], targets[0], end='\r')
        # print(f'{loss}    {epi}    ', end='\r')

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network every few steps
        if epi % self.TARGET_UPDATE_FREQ == 0:
            self.update_target()

        return loss.item()
    
    def optimizer_to_dev(self):
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)

    def train(self):
        # epsilon greedy exploration

        start_time = time.time()
        self.policy_net = self.policy_net.to(self.device)
        self.target_net = self.target_net.to(self.device)
        self.optimizer_to_dev()

        for epoch in tqdm(range(self.num_epoch)):
            self.state = env.reset()
            self.state = torch.tensor(self.state, dtype=torch.float32, device=self.device).unsqueeze(0)
            self.contact_made = False
            logging.info(f'Epoch {self.epoch}')

            # actions = []
            epi = 0
            # for frame in tqdm(range(100000)):
            for frame in count():

                
                if frame % self.action_freq == 0:
                    # Play an episode and log episodic reward
                    self.action = self.policy()
                    observation, _, done, _ = env.step(env.available_actions[self.action])
                    # actions.append(action.item())
                    if done:
                        self.next_state = None
                    else:
                        self.next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                    epi += 1
                
                elif frame % self.action_freq == self.action_freq - 1:
                    # Store the transition in memory after reward has been accumulated
                    _, reward, _, _ = env.step(None)
                    reward = torch.tensor([reward], device=self.device)
                    self.memory.push(self.state, self.action, self.next_state, reward)

                    self.state = self.next_state
                
                    # Train after collecting sufficient experience
                    if len(self.memory) >= self.BATCH_SIZE*5:
                        self.update_networks(epi)

                else:
                    env.step(None)

                cur_time = time.time()
                if cur_time - start_time > self.checkpoint_interval:
                    self.commit_state()
                    start_time = cur_time
                
                if env.is_pushing:
                    self.last_epi_contact_made = epi

                if epi > self.last_epi_contact_made + 2000:
                    done = True
                    logging.info("No contact made. Resetting environment...")
                    # action_path = os.path.join(os.path.dirname(self.checkpoint_path), "actions.pt")
                    # torch.save(actions, action_path)

                if done:
                    if epi <= self.last_epi_contact_made + 2000:
                        logging.info("Object in receptacle. Resetting environment...")
                    break

            self.epoch += 1
    

def main(args):
    global env
    if args.log_file is not None:
        logging.basicConfig(filename=args.log_file,level=logging.DEBUG)
    
    logging.info("starting training script")

    env = environments.selector(args.environment)
    env.state_type = args.state_type
    

    train = Train_DQL(args.state_type, args.model, args.checkpoint_path, args.checkpoint_interval, args.num_epoch, args.batch_size)
    
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
        default= 'vision'
    )

    parser.add_argument(
        '--model',
        type=str,
        help='options: ["resnet", "densenet"]',
        default='resnet'
    )

    parser.add_argument(
        '--num_epoch',
        type=int,
        help='number of epochs to run',
        default=50
    )

    parser.add_argument(
        '--checkpoint_path',
        type=str,
        help='path to save and look for the checkpoint file',
        default=os.path.join(os.getcwd(), "checkpoint/checkpoint.pt")
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        help='batch size per iteration',
        default=64
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
        default=os.path.join(os.getcwd(), "train.log")
    )

    parser.add_argument(
        '--environment',
        type=int,
        help='environment to simulate- 0: nav_obstacle, 1: push_empty, 2: push_empty_small',
        default=2
    )

    main(parser.parse_args())