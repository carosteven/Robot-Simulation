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
                        ('state', 'action', 'next_state', 'reward', 'distance'))



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
        self.options = config['options']
        self.num_policies = config['num_policies']
        self.checkpoint_path = config['checkpoint_path']
        self.checkpoint_interval = config['checkpoint_interval']
        self.no_goal_timeout = config['no_goal_timeout']
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
        
        self.policies = []
        for i in range(self.num_policies):
            policy = self.create_or_restore_training_state(config['state_type'], config['model'], config['replay_buffer_size'], hierarchy=i)
            self.policies.append(policy)

        if self.options:
            self.policies[0]['n_actions'] = 3
            self.policies[1]['n_actions'] = env.screen_size[0]*env.screen_size[1]
        else:
            self.policies[0]['n_actions'] = env.screen_size[0]*env.screen_size[1]
        
        self.steps_done = 0 # for exploration
        self.contact_made = False # end episode if agent does not push box after x actions
        self.last_epi_box_in_goal = 0

    def create_or_restore_training_state(self, state_type, model, buffer_size, hierarchy=0):
        if self.options and hierarchy == 0:
            policy_net = models.VisionDQN(self.n_observations, n_actions=3)
            target_net = models.VisionDQN(self.n_observations, n_actions=3)
        else:
            if state_type == 'vision':
                if model == 'resnet':
                    if self.action_type == 'straight-line-navigation':
                        policy_net = models.VisionDQN_SAM(self.n_observations)
                        target_net = models.VisionDQN_SAM(self.n_observations)
                    else:
                        policy_net = models.VisionDQN(self.n_observations, self.n_actions)
                        target_net = models.VisionDQN(self.n_observations, self.n_actions)

                elif model == 'densenet':
                    policy_net = models.VisionDQN_dense(self.n_observations, self.n_actions)
                    target_net = models.VisionDQN_dense(self.n_observations, self.n_actions)
                
            else:    
                policy_net = models.SensorDQN(self.n_observations, self.n_actions)
                target_net = models.SensorDQN(self.n_observations, self.n_actions)
        target_net.load_state_dict(policy_net.state_dict())

        # self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.LEARNING_RATE)
        # self.optimizer = optim.SGD(self.policy_net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
        optimizer = optim.AdamW(policy_net.parameters(), lr=self.LEARNING_RATE, weight_decay=0.01)
        memory = ReplayMemory(buffer_size)
        self.epoch = 0
        loss = 0
        epsilon = self.STARTING_EPSILON

        if os.path.exists(self.checkpoint_path):
            training_state = torch.load(self.checkpoint_path)
            # print(training_state[f'policy_state_dict_0'])
            self.epoch = training_state['epoch']
            policy_net.load_state_dict(training_state[f'policy_state_dict_{hierarchy}'])
            target_net.load_state_dict(training_state[f'target_state_dict_{hierarchy}'])
            optimizer.load_state_dict(training_state[f'optimizer_state_dict_{hierarchy}'])
            memory.memory = training_state[f'memory_{hierarchy}']
            loss = training_state[f'loss_{hierarchy}']
            epsilon = training_state[f'epsilon_{hierarchy}']
            logging.info(f"Training state restored at epoch {self.epoch}")
        else:
            logging.info("No checkpoint detected, starting from initial state")

        return {'policy_net': policy_net, 'target_net': target_net, 'optimizer': optimizer, 'memory': memory, 'loss': loss, 'epsilon': epsilon}

    def commit_state(self):
        temp_path = os.path.join(os.path.dirname(self.checkpoint_path), "temp.pt")
        training_state = {}
        training_state['epoch'] = self.epoch
        for i in range(self.num_policies):
            training_state[f'policy_state_dict_{i}'] = self.policies[i]['policy_net'].state_dict()
            training_state[f'target_state_dict_{i}'] = self.policies[i]['target_net'].state_dict()
            training_state[f'optimizer_state_dict_{i}'] = self.policies[i]['optimizer'].state_dict()
            training_state[f'memory_{i}'] = self.policies[i]['memory'].memory
            training_state[f'loss_{i}'] = self.policies[i]['loss']
            training_state[f'epsilon_{i}'] = self.policies[i]['epsilon']

        # first save the temp file
        torch.save(training_state, temp_path)
        # according to the GNU spec of rename, the state of checkpoint_path
        # is atomic, i.e. it will either be modified or not modified, but not in
        # between, during a system crash (i.e. preemtion)
        os.replace(temp_path, self.checkpoint_path)
        msg = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ": Checkpoint saved at " + self.checkpoint_path
        logging.info(msg)

        # Update a target network using a source network
    def update_target(self, policy):
        for tp, p in zip(policy['target_net'].parameters(), policy['policy_net'].parameters()):
            tp.data.copy_(p.data)
        return policy['target_net']
    
    def get_action(self, policy):
        # With probability EPSILON, choose a random action
        # Rest of the time, choose argmax_a Q(s, a) 
        if np.random.rand() < policy['epsilon']:
            if policy['n_actions'] > 16:
                action = np.random.randint(policy['n_actions']/4) # /4 because the screen is 304x304 but the action space is 152x152
            else:
                action = np.random.randint(policy['n_actions'])

        else:
            qvalues = policy['policy_net'](self.transform_state(self.state))
            action = torch.argmax(qvalues).item()

        action = torch.tensor([[action]], device=self.device, dtype=torch.long)
        
        # Epsilon update rule: Keep reducing a small amount over
        # STEPS_MAX number of steps, and at the end, fix to EPSILON_END
        prev_eps = policy['epsilon']
        policy['epsilon'] = max(self.EPSILON_END, policy['epsilon'] - (1.0 / self.STEPS_MAX))
        if policy['epsilon'] == self.EPSILON_END and policy['epsilon'] != prev_eps:
            logging.info("Reached min epsilon")

        return action
    
    def transform_state(self, state_batch): #TODO check if transforming state before storing in memory is more efficient
        colour_batch = torch.zeros((state_batch.shape[0], 3, state_batch.shape[2], state_batch.shape[3]),device=self.device)
        # print(state_batch[:,0])
        colour_batch[:,0] = torch.bitwise_right_shift(state_batch[:,0], 16)
        colour_batch[:,1] = torch.bitwise_right_shift(state_batch[:,0], 8&255)
        colour_batch[:,2] = torch.bitwise_and(state_batch[:,0], 255)
        return colour_batch
    
    def update_networks(self, policy, epi):
    
        # Sample a minibatch (s, a, r, s', d)
        # Each variable is a vector of corresponding values
        transitions = policy['memory'].sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = self.transform_state(torch.cat([s for s in batch.next_state if s is not None]))
        state_batch = self.transform_state(torch.cat(batch.state)).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        distance_batch = torch.tensor(batch.distance, device=self.device, dtype=torch.float)

        # Get Q(s, a) for every (s, a) in the minibatch
        qvalues = policy['policy_net'](state_batch).gather(1, action_batch.view(-1, 1)).squeeze()
        
        # Double DQN Formula: r + gamma*TARGET(s_t+1, argmax_a POLICY (s_t+1, a))
        q_target_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            actions = torch.argmax(policy['policy_net'](non_final_next_states), dim=1)
            q_target_values[non_final_mask] = policy['target_net'](non_final_next_states).gather(1, actions.unsqueeze(1)).squeeze()
        targets = reward_batch + torch.pow(self.GAMMA, distance_batch) * q_target_values

        # Detach y since it is the target. Target values should
        # be kept fixed.
        loss = torch.nn.SmoothL1Loss()(targets.detach(), qvalues)

        # Backpropagation
        policy['optimizer'].zero_grad()
        loss.backward()
        policy['optimizer'].step()

        # Update target network every few steps
        if epi % self.TARGET_UPDATE_FREQ == 0: #NOTE epi is updated after every option so more frequently than every sln action (so not really correct)
            policy['target_policy'] = self.update_target(policy)

        return loss.item()
    
    def optimizer_to_dev(self, optimizer):
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)
        return optimizer

    def train(self):

        start_time = time.time()
        for i in range(self.num_policies):
            self.policies[i]['policy_net'] = self.policies[i]['policy_net'].to(self.device)
            self.policies[i]['target_net'] = self.policies[i]['target_net'].to(self.device)
            self.policies[i]['optimizer']  = self.optimizer_to_dev(self.policies[i]['optimizer'])

        for epoch in tqdm(range(self.num_epochs)):
            # Reset environment and get new state
            self.state = torch.tensor(env.reset(), dtype=torch.int32, device=self.device).unsqueeze(0)
            self.contact_made = False
            logging.info(f'Epoch {self.epoch}')

            # actions = []
            epi = 0
            done = False
            # for frame in tqdm(range(100000)):
            for frame in count():
                if self.options:
                    option = self.get_action(self.policies[0])
                    if option == 0:
                        reward, epi, done = self.primitive_action_control(None, frame, epi, action=0) # 0 is forward

                    elif option == 1:
                        reward, epi, done = self.primitive_action_control(None, frame, epi, action=1) # 1 is backward

                    elif option == 2:
                        reward, epi, done = self.sln_action_control(self.policies[1], frame, epi)
                    
                    self.policies[0]['memory'].push(self.state, option, self.next_state, reward, 1)

                    if len(self.policies[0]['memory']) >= self.BATCH_SIZE*self.num_of_batches_before_train:
                        self.update_networks(self.policies[0], epi)

                else:
                    if self.action_type == 'primitive':
                        _, epi, done = self.primitive_action_control(self.policies[0], frame, epi)

                    elif self.action_type == 'straight-line-navigation':
                        _, epi, done = self.sln_action_control(self.policies[0], frame, epi)

                self.state = self.next_state

                cur_time = time.time()
                if cur_time - start_time > self.checkpoint_interval:
                    self.commit_state()
                    start_time = cur_time
                
                # if env.is_pushing:
                #     self.last_epi_box_in_goal = epi

                if epi > self.last_epi_box_in_goal + self.no_goal_timeout:
                    done = True
                    logging.info(f"Inactivity timeout. {env.config['num_boxes'] - env.boxes_remaining} in goal. Resetting environment...")

                if done:
                    if epi <= self.last_epi_box_in_goal + self.no_goal_timeout:
                        logging.info("All boxes in receptacle. Resetting environment...")
                    break

            self.epoch += 1

    def sln_action_control(self, policy, frame, epi):
        self.action = self.get_action(policy)
        action = np.unravel_index(self.action[0,0].cpu(), (int(env.screen_size[0]/2), int(env.screen_size[1]/2)))
        action = (action[0]*2, action[1]*2)

        total_reward = 0
        while not env.action_completed:
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            if env.boxes_in_goal != 0:
                logging.info(f"{env.boxes_in_goal} boxes added to receptacle.")
                self.last_epi_box_in_goal = epi
        env.action_completed = False

        if done:
            self.next_state = None
        else:
            self.next_state = torch.tensor(next_state, dtype=torch.int32, device=self.device).unsqueeze(0)
            
        total_reward = torch.tensor([total_reward], device=self.device)
        policy['memory'].push(self.state, self.action, self.next_state, total_reward, info['distance'])

        # self.state = self.next_state #############
    
        # Train after collecting sufficient experience
        if len(policy['memory']) >= self.BATCH_SIZE*self.num_of_batches_before_train:
            self.update_networks(policy, epi)

        epi += 1
        return total_reward, epi, done

    def primitive_action_control(self, policy, frame, epi, action=None):
        if action is not None:
            total_reward = 0
            for frame in range(self.action_freq):
                if frame == 0:
                    _, reward, done, _ = env.step(env.available_actions[action], primitive=True)

                elif frame == self.action_freq - 1:
                    env.action_completed = True
                    next_state, reward, done, _ = env.step(None, primitive=True)
                    if done:
                        self.next_state = None
                    else:
                        self.next_state = torch.tensor(next_state, dtype=torch.int32, device=self.device).unsqueeze(0)
                        
                    # reward = torch.tensor([reward], device=self.device)

                else:
                    _, reward, done, _ = env.step(None, primitive=True)
                
                total_reward += reward
                if done:
                    self.next_state = None
                    break
            
            env.action_completed = False
            epi += 1
            total_reward = torch.tensor([total_reward], device=self.device)
            return total_reward, epi, done


        if frame % self.action_freq == 0:
            # Play an episode and log episodic reward
            self.action = self.get_action(policy)
            env.step(env.available_actions[self.action], primitive=True)

            epi += 1
        
        elif frame % self.action_freq == self.action_freq - 1:
            # Store the transition in memory after reward has been accumulated
            next_state, reward, done, _ = env.step(None, primitive=True)
            if done:
                self.next_state = None
            else:
                self.next_state = torch.tensor(next_state, dtype=torch.int32, device=self.device).unsqueeze(0)
                
            reward = torch.tensor([reward], device=self.device)
            policy['memory'].push(self.state, self.action, self.next_state, reward)

            # self.state = self.next_state
        
            # Train after collecting sufficient experience
            if len(policy['memory']) >= self.BATCH_SIZE*self.num_of_batches_before_train:
                self.update_networks(epi)

        else:
            env.step(None, primitive=True)
        
        return epi
    

def main(args):
    with open(args.config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    global env
    if config['log_file'] is not None:
        logging.basicConfig(filename=config['log_file'],level=logging.DEBUG)
    
    logging.info("starting training script")

    env = environments.selector(config)
    env.state_type = config['state_type']
    '''
    if config['action_type'] == 'primitive':
        env.take_action = env._actions
    elif config['action_type'] == 'straight-line-navigation':
        env.take_action = env.straight_line_navigation
    '''

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
        # default= 'configurations/config_test.yml'
        default= 'configurations/config_push_small_sln_options.yml'
    )

    main(parser.parse_args())