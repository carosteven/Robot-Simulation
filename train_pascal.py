import matplotlib.pyplot as plt
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
import yaml

import logging
logging.getLogger('pymunk').propagate = False

import environments
import models

from PIL import Image

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
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_type = config['action_type']
        self.checkpoint_path = config['checkpoint_path']
        self.checkpoint_interval = config['checkpoint_interval']
        self.num_epochs = config['num_epochs']
        self.num_of_batches_before_train = config['num_of_batches_before_train']
        # Get number of actions from env
        self.n_actions = len(env.available_actions) if config['action_type'] == 'primitive' else env.screen_size[0]*env.screen_size[1]

        self.action_freq = 25

        # Global variables
        self.BATCH_SIZE = config['batch_size']                  # How many examples to sample per train step
        self.GAMMA = config['gamma']                            # Discount factor in episodic reward objective
        self.LEARNING_RATE = config['lr']                       # Learning rate for Adam optimizer
        self.TARGET_UPDATE_FREQ = config['target_update_freq']  # Target network update frequency
        self.STARTING_EPSILON = config['epsilon_start']         # Starting epsilon
        self.STEPS_MAX = config['epsilon_steps']                # Gradually reduce epsilon over these many steps
        self.EPSILON_END = config['epsilon_end']                # At the end, keep epsilon at this value

        self.EPSILON = self.STARTING_EPSILON

        # Get number of state observations
        self.state = torch.tensor(env.reset(), dtype=torch.int32, device=self.device).unsqueeze(0)
        # self.n_observations = len(self.state)
        self.n_observations = 3 # (channels)
        self.next_state = None
        self.action = None
        
        self.create_or_restore_training_state(config['state_type'], config['model'], config['replay_buffer_size'])

        self.steps_done = 0 # for exploration
        self.contact_made = False # end episode if agent does not push box after x actions
        self.last_epi_contact_made = 0

    def create_or_restore_training_state(self, state_type, model, buffer_size):
        if state_type == 'vision':
            if model == 'resnet':
                if self.action_type == 'straight-line-navigation':
                    self.policy_net = models.VisionDQN_SAM(self.n_observations)
                    self.target_net = models.VisionDQN_SAM(self.n_observations)
                else:
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
        self.memory = ReplayMemory(buffer_size)
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
            action = np.random.randint(self.n_actions/4) # /4 because the screen is 304x304 but the action space is 152x152
        else:
            qvalues = self.policy_net(self.transform_state(self.state))
            action = torch.argmax(qvalues).item()

        action = torch.tensor([[action]], device=self.device, dtype=torch.long)
        
        # Epsilon update rule: Keep reducing a small amount over
        # STEPS_MAX number of steps, and at the end, fix to EPSILON_END
        prev_eps = self.EPSILON
        self.EPSILON = max(self.EPSILON_END, self.EPSILON - (1.0 / self.STEPS_MAX))
        if self.EPSILON == self.EPSILON_END and self.EPSILON != prev_eps:
            logging.info("Reached min epsilon")

        return action
    
    def transform_state(self, state_batch):
        colour_batch = torch.zeros((state_batch.shape[0], 3, state_batch.shape[2], state_batch.shape[3]),device=self.device)
        colour_batch[:,0] = torch.bitwise_right_shift(state_batch[:,0], 16)
        colour_batch[:,1] = torch.bitwise_right_shift(state_batch[:,0], 8&255)
        colour_batch[:,2] = torch.bitwise_and(state_batch[:,0], 255)
        return colour_batch
    
    def update_networks(self, epi):
    
        # Sample a minibatch (s, a, r, s', d)
        # Each variable is a vector of corresponding values
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = self.transform_state(torch.cat([s for s in batch.next_state if s is not None]))
        state_batch = self.transform_state(torch.cat(batch.state))
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
        # TODO why are q2values zero when the state is final?
        q_target_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            actions = torch.argmax(self.policy_net(non_final_next_states), dim=1)
            q_target_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, actions.unsqueeze(1)).squeeze()
        targets = reward_batch + self.GAMMA * q_target_values

        # Detach y since it is the target. Target values should
        # be kept fixed.
        loss = torch.nn.SmoothL1Loss()(targets.detach(), qvalues)

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

        start_time = time.time()
        self.policy_net = self.policy_net.to(self.device)
        self.target_net = self.target_net.to(self.device)
        self.optimizer_to_dev()

        for epoch in tqdm(range(self.num_epoch)):
            # Reset environment and get new state
            self.state = torch.tensor(env.reset(), dtype=torch.int32, device=self.device).unsqueeze(0)
            self.contact_made = False
            logging.info(f'Epoch {self.epoch}')

            # actions = []
            epi = 0
            done = False
            # for frame in tqdm(range(100000)):
            for frame in count():
                if self.action_type == 'primitive':
                    epi, done = self.primitive_action_control(frame, epi)

                elif self.action_type == 'straight-line-navigation':
                    epi, done = self.sln_action_control(frame, epi)

                cur_time = time.time()
                if cur_time - start_time > self.checkpoint_interval:
                    self.commit_state()
                    start_time = cur_time
                
                if env.is_pushing:
                    self.last_epi_contact_made = epi

                if epi > self.last_epi_contact_made + 2000: # Make this a parameter
                    done = True
                    logging.info("No contact made. Resetting environment...")

                if done:
                    if epi <= self.last_epi_contact_made + 2000:
                        logging.info("Object in receptacle. Resetting environment...")
                    break

            self.epoch += 1

    def sln_action_control(self, frame, epi):
        self.action = self.policy()
        action = np.unravel_index(self.action[0,0].cpu(), (int(env.screen_size[0]/2), int(env.screen_size[1]/2)))
        action = (action[0]*2, action[1]*2)

        while not env.action_completed:
            observation, reward, done, _ = env.step(action)
        env.action_completed = False

        if done:
            self.next_state = None
        else:
            self.next_state = torch.tensor(observation, dtype=torch.int32, device=self.device).unsqueeze(0)
            
        reward = torch.tensor([reward], device=self.device)
        self.memory.push(self.state, self.action, self.next_state, reward)

        self.state = self.next_state
    
        # Train after collecting sufficient experience
        if len(self.memory) >= self.BATCH_SIZE*self.num_of_batches_before_train:
            self.update_networks(epi)

        epi += 1
        return epi, done

    def primitive_action_control(self, frame, epi):
        if frame % self.action_freq == 0:
            # Play an episode and log episodic reward
            self.action = self.policy()
            env.step(env.available_actions[self.action])

            epi += 1
        
        elif frame % self.action_freq == self.action_freq - 1:
            # Store the transition in memory after reward has been accumulated
            observation, reward, done, _ = env.step(None)
            if done:
                self.next_state = None
            else:
                self.next_state = torch.tensor(observation, dtype=torch.int32, device=self.device).unsqueeze(0)
                
            reward = torch.tensor([reward], device=self.device)
            self.memory.push(self.state, self.action, self.next_state, reward)

            self.state = self.next_state
        
            # Train after collecting sufficient experience
            if len(self.memory) >= self.BATCH_SIZE*self.num_of_batches_before_train:
                self.update_networks(epi)

        else:
            env.step(None)
        
        return epi
    

def main(args):
    with open(args.config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    global env
    if config['log_file'] is not None:
        logging.basicConfig(filename=config['log_file'],level=logging.DEBUG)
    
    logging.info("starting training script")

    env = environments.selector(config['environment'])
    env.state_type = config['state_type']
    if config['action_type'] == 'primitive':
        env.take_action = env._actions
    elif config['action_type'] == 'sln':
        env.take_action = env.straight_line_navigation
    

    train = Train_DQL(config)
    
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
        '--config_file',
        type=str,
        help='path of the configuration file',
        default= 'configurations/config_test.yml'
    )

    main(parser.parse_args())