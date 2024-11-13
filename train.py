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

import environments
import models

from PIL import Image

logging.getLogger('pymunk').propagate = False

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
    def __init__(self, config, test):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_type = config['action_type']
        self.options = config['options']
        self.num_policies = config['num_policies']
        self.checkpoint_path = config['checkpoint_path'] if not test else config['model_path']
        print(self.checkpoint_path.split('/')[-1])
        self.checkpoint_interval = config['checkpoint_interval']
        self.no_goal_timeout = config['no_goal_timeout']
        self.num_epochs = config['num_epochs']
        self.num_of_batches_before_train = config['num_of_batches_before_train']
        self.test = test
        # Get number of actions from env
        self.n_actions = len(env.available_actions) if config['action_type'] == 'primitive' else env.screen_size[0]*env.screen_size[1]

        self.action_freq = config['action_freq']
        self.state_info = config['state_info']

        # Global variables
        self.BATCH_SIZE = config['batch_size']                  # How many examples to sample per train step
        self.GAMMA = config['gamma']                            # Discount factor in episodic reward objective
        self.LEARNING_RATE = config['lr']                       # Learning rate for Adam optimizer
        self.TARGET_UPDATE_FREQ = config['target_update_freq']  # Target network update frequency
        self.STARTING_EPSILON = config['epsilon_start']         # Starting epsilon
        self.STEPS_MAX = config['epsilon_steps']                # Gradually reduce epsilon over these many steps
        self.EPSILON_END = config['epsilon_end']                # At the end, keep epsilon at this value

        self.EPSILON = self.STARTING_EPSILON

        if self.state_info == 'colour':
            self.transform_state = self.trans_colour_state
            # Get number of state observations
            # self.n_observations = len(self.state)
            self.n_observations = 3 # (channels)

        elif self.state_info == 'multiinfo':
            self.transform_state = self.trans_multiinfo_state
            self.n_observations = 4 # (channels)

        self.state = self.get_state(env.reset())
        self.next_state = None
        self.action = None
        
        self.episodic_stats = {'cumulative_reward': [], 'num_steps': [], 'boxes_in_goal': []}
        
        self.policies = []
        for i in range(self.num_policies):
            policy = self.create_or_restore_training_state(config['state_type'], config['model'], config['replay_buffer_size'], hierarchy=i)
            self.policies.append(policy)

        if self.options:
            self.policies[0]['n_actions'] = 3
            self.policies[1]['n_actions'] = env.screen_size[0]*env.screen_size[1]
        else:
            self.policies[0]['n_actions'] = self.n_actions
        
        self.steps_done = 0 # for exploration
        self.contact_made = False # end episode if agent does not push box after x actions
        self.last_epi_box_in_goal = 0

    
    def get_state(self, raw_state):
        if self.state_info == 'colour':
            state = torch.tensor(raw_state, dtype=torch.int32, device=self.device).unsqueeze(0)
        elif self.state_info == 'multiinfo':
            state = []
            state.append(torch.tensor(raw_state[0], dtype=torch.int32, device=self.device).unsqueeze(0))
            # NOTE this is a hack to get the agent position into the state tensor
            # because the state tensor has to be the same shape. Could save a lot of storage by not doing this
            state.append(torch.zeros_like(state[0], device=self.device, dtype=torch.int32))
            state[1][0,0,0,0] = raw_state[1][0]
            state[1][0,0,0,1] = raw_state[1][1]
            state = torch.stack(state, dim=1)
        return state


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
            training_state = torch.load(self.checkpoint_path, map_location=self.device)
            # Remove the last element from the stats since it is not complete
            for key in training_state['stats'].keys():
                if len(training_state['stats'][key]) > 0:
                    training_state['stats'][key].pop()
            self.episodic_stats = training_state['stats']

            if self.test:
                policy_net.load_state_dict(training_state[f'policy_state_dict_{hierarchy}'])
                self.show_stats(self.episodic_stats)
                epsilon = self.EPSILON_END
            else:
                training_state = torch.load(self.checkpoint_path)
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
        training_state['stats'] = self.episodic_stats
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
    
    def trans_colour_state(self, state_batch):
        colour_batch = torch.zeros((state_batch.shape[0], 3, state_batch.shape[2], state_batch.shape[3]),device=self.device,dtype=torch.float32)
        colour_batch[:,0] = torch.bitwise_and(torch.bitwise_right_shift(state_batch[:,0], 16), 255) # Red channel
        colour_batch[:, 1] = torch.bitwise_and(torch.bitwise_right_shift(state_batch[:, 0], 8), 255)  # Green channel
        colour_batch[:, 2] = torch.bitwise_and(state_batch[:, 0], 255)  # Blue channel
        colour_batch = colour_batch / 255.0
        return colour_batch
    
    def trans_multiinfo_state(self, state_batch):
        # Returns a tensor of shape (batch_size, 4, height, width)
        # Channel 0: Grayscale image
        # Channel 1: Mask of the agent
        # Channel 2: Distance from the agent to every pixel
        # Channel 3: Distance from the goal to every pixel
        # state_batch is a tensor of shape (batch_size, 2)
        # where the first element is an integer rgb array of the environment and the second element is the position of the agent
        
        # Extract the rgb array and the agent position
        image = state_batch[:, 0, 0]
        agent_pos = state_batch[:,1,0,0,0:2]/2
        agent_pos = agent_pos.int()

        # Get the height and width of the image
        height, width = image.shape[1], image.shape[2]

        # Create a tensor of shape (batch_size, 4, height, width) filled with zeros
        multiinfo_batch = torch.zeros((state_batch.shape[0], 4, height, width), device=self.device, dtype=torch.float32)

        # Extract the red, green, and blue channels from the rgb array
        red = torch.bitwise_and(torch.bitwise_right_shift(image, 16), 255)
        green = torch.bitwise_and(torch.bitwise_right_shift(image, 8), 255)
        blue = torch.bitwise_and(image, 255)

        # Calculate the grayscale image
        grayscale = 0.2989 * red + 0.5870 * green + 0.1140 * blue

        # Calculate the mask of the agent (28 pixels (scaled so 14) around the agent)
        agent_mask = torch.zeros((state_batch.shape[0], height, width), device=self.device, dtype=torch.float32)
        agent_mask = torch.zeros((state_batch.shape[0], height, width), device=self.device, dtype=torch.float32)
        y_indices, x_indices = torch.meshgrid(torch.arange(height, device=self.device), torch.arange(width, device=self.device), indexing='ij')
        y_indices = y_indices.unsqueeze(0).expand(state_batch.shape[0], -1, -1)
        x_indices = x_indices.unsqueeze(0).expand(state_batch.shape[0], -1, -1)
        
        agent_mask = ((x_indices >= (agent_pos[:, 0].unsqueeze(1).unsqueeze(2) - 7)) & 
                  (x_indices < (agent_pos[:, 0].unsqueeze(1).unsqueeze(2) + 7)) & 
                  (y_indices >= (agent_pos[:, 1].unsqueeze(1).unsqueeze(2) - 7)) & 
                  (y_indices < (agent_pos[:, 1].unsqueeze(1).unsqueeze(2) + 7))).float()

        # Calculate the distance from the agent to every pixel
        y_indices, x_indices = torch.meshgrid(torch.arange(height, device=self.device), torch.arange(width, device=self.device), indexing='ij')
        y_indices, x_indices = y_indices.unsqueeze(0), x_indices.unsqueeze(0)
        agent_distance = torch.sqrt((x_indices - agent_pos[:, 0].unsqueeze(1).unsqueeze(2))**2 + (y_indices - agent_pos[:, 1].unsqueeze(1).unsqueeze(2))**2)

        # Calculate the distance from the goal to every pixel
        goal_distance = torch.zeros((state_batch.shape[0], height, width), device=self.device, dtype=torch.float32)
        goal_position = torch.tensor(env.goal_position, device=self.device, dtype=torch.float32) / 2
        goal_x, goal_y = goal_position[0], goal_position[1]
        goal_distance = torch.sqrt((x_indices - goal_x)**2 + (y_indices - goal_y)**2)

        # to visualize the multiinfo tensor
        # fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        # axs[0].imshow(grayscale[0].cpu().numpy(), cmap='gray')
        # axs[0].set_title('Grayscale')
        # axs[1].imshow(agent_mask[0].cpu().numpy(), cmap='gray')
        # axs[1].set_title('Agent Mask')
        # axs[2].imshow(agent_distance[0].cpu().numpy(), cmap='hot')
        # axs[2].set_title('Agent Distance')
        # axs[3].imshow(goal_distance[0].cpu().numpy(), cmap='hot')
        # axs[3].set_title('Goal Distance')
        # plt.show()
        # input()

        # Fill the multiinfo tensor with the grayscale image, the agent mask, the agent distance, and the goal distance
        multiinfo_batch[:, 0] = grayscale / 255.0
        multiinfo_batch[:, 1] = agent_mask
        multiinfo_batch[:, 2] = agent_distance / (304.0/2)
        multiinfo_batch[:, 3] = goal_distance / (304.0/2)

        return multiinfo_batch
    
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
        loss = torch.nn.SmoothL1Loss()(targets.detach().view_as(qvalues), qvalues)

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
    
    def scheduler(self):
        success_threshold = 0.8  # Define a success threshold
        evaluation_interval = 10  # Define window to evaluate success rate
        success_count = sum(1 for boxes in self.episodic_stats['boxes_in_goal'][-evaluation_interval:] if boxes == env.training_step+1)
        success_rate = success_count / evaluation_interval

        if success_rate >= success_threshold:
            self.increase_difficulty()


    def increase_difficulty(self):
        # I think in "Curriculum RL From Avoiding Collision..." they do not clear the replay buffer
        # env.config['num_boxes'] -= 5
        # logging.info(f"\nIncreasing difficulty: grid_size={env.config['grid_size']}, num_boxes={env.config['num_boxes']}\n")
        if env.training_step <= 3:
            logging.info(f"\nIncreasing difficulty\n")
            if env.training_step == 3:
                logging.info(f"\nReached maximum difficulty\n")
            env.training_step += 1
            env.config['num_boxes'] -= 5
            for policy in self.policies:
                policy['epsilon'] = self.STARTING_EPSILON
        
        # else:
        #     self.box_positions = [(x, y) for x in range(env.grid_size-1) for y in range(env.grid_size-2, -1, -1) if (x, y) not in [(0, 0), (0, 1), (1, 0), (1, 1)]]

    def train(self):
        start_time = time.time()
        for i in range(self.num_policies):
            self.policies[i]['policy_net'] = self.policies[i]['policy_net'].to(self.device)
            if not self.test:
                self.policies[i]['target_net'] = self.policies[i]['target_net'].to(self.device)
                self.policies[i]['optimizer']  = self.optimizer_to_dev(self.policies[i]['optimizer'])

        for epoch in tqdm(range(self.num_epochs)):
            # Reset environment and get new state
            self.scheduler()
            self.state = self.get_state(env.reset())
            self.contact_made = False
            logging.info(f'Epoch {self.epoch}')

            # Keep track of stats
            self.episodic_stats['cumulative_reward'].append(0)
            self.episodic_stats['num_steps'].append(0)
            self.episodic_stats['boxes_in_goal'].append(0)

            # actions = []
            epi = 0
            self.last_epi_box_in_goal = 0
            done = False
            timeout = False
            # for frame in tqdm(range(100000)):
            for frame in count():
                if epi > self.last_epi_box_in_goal + self.no_goal_timeout:
                    timeout = True

                if self.options:
                    option = self.get_action(self.policies[0])
                    if option == 0:
                        reward, epi, done = self.primitive_action_control(None, frame, epi, action=0) # 0 is forward

                    elif option == 1:
                        reward, epi, done = self.primitive_action_control(None, frame, epi, action=1) # 1 is backward

                    elif option == 2:
                        reward, epi, done = self.sln_action_control(self.policies[1], frame, epi)
                    
                    if not self.test:
                        self.policies[0]['memory'].push(self.state, option, self.next_state, reward, 1)

                        if len(self.policies[0]['memory']) >= self.BATCH_SIZE*self.num_of_batches_before_train:
                            self.update_networks(self.policies[0], epi)

                else:
                    if self.action_type == 'primitive':
                        reward, epi, done = self.primitive_action_control(self.policies[0], frame, epi, timeout=timeout)

                    elif self.action_type == 'straight-line-navigation':
                        _, epi, done = self.sln_action_control(self.policies[0], frame, epi)

                self.state = self.next_state

                # Update stats
                self.episodic_stats['cumulative_reward'][-1] += reward.item()
                self.episodic_stats['num_steps'][-1] = epi
                self.episodic_stats['boxes_in_goal'][-1] = env.config['num_boxes'] - env.boxes_remaining

                cur_time = time.time()
                if cur_time - start_time > self.checkpoint_interval and not self.test:
                    self.commit_state()
                    start_time = cur_time
                
                # if env.is_pushing:
                #     self.last_epi_box_in_goal = epi

                if done or timeout:
                    if timeout:
                        logging.info(f"Inactivity timeout. {env.config['num_boxes'] - env.boxes_remaining} in goal. Resetting environment...")
                    if done:
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

    def primitive_action_control(self, policy, frame, epi, action=None, timeout=False):
        if action is not None:
            total_reward = 0
            for frame in range(self.action_freq):
                if frame == 0:
                    _, reward, done, _ = env.step(env.available_actions[action], primitive=True)

                elif frame == self.action_freq - 1:
                    env.action_completed = True
                    next_state, reward, done, _ = env.step(None, primitive=True)
                    if done: # or timeout
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
            if env.boxes_in_goal != 0:
                logging.info(f"{env.boxes_in_goal} boxes added to receptacle.")
                self.last_epi_box_in_goal = epi
            return total_reward, epi, done

        if frame % self.action_freq == 0:
            prev_boxes_in_goal = env.boxes_in_goal
            # Play an episode and log episodic reward
            self.action = self.get_action(policy)
            env.step(env.available_actions[self.action], primitive=True)
            if self.test:
                print(env.available_actions[self.action])

            epi += 1
        
        if frame % self.action_freq == self.action_freq - 1:
            # Store the transition in memory after reward has been accumulated
            next_state, reward, done, info = env.step(None, primitive=True)
            if done or timeout:
                # self.episodic_stats['cumulative_reward'].append(info['cumulative_reward'])
                self.next_state = None
            else:
                self.next_state = self.get_state(next_state)
            reward = torch.tensor([reward], device=self.device)
            # print(env.boxes_in_goal)
            if env.boxes_in_goal > prev_boxes_in_goal:
                logging.info(f"{env.boxes_in_goal - prev_boxes_in_goal} boxes added to receptacle.")
                self.last_epi_box_in_goal = epi
            
            if not self.test:
                policy['memory'].push(self.state, self.action, self.next_state, reward, 1)

                # Train after collecting sufficient experience
                if len(policy['memory']) >= self.BATCH_SIZE*self.num_of_batches_before_train:
                    self.update_networks(policy, epi)

        elif frame % self.action_freq != 0:
            env.step(None, primitive=True)
        # print(reward)
        return reward, epi, done
    
    def show_stats(self, stats):
        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()
        ax3 = ax1.twinx()

        ax3.spines['right'].set_position(('outward', 60))

        ax1.plot(stats['cumulative_reward'], 'g-', label='Cumulative Reward')
        ax2.plot(stats['num_steps'], 'b-', label='Number of Steps')
        ax3.bar(range(len(stats['boxes_in_goal'])), stats['boxes_in_goal'], alpha=0.3, color='r', label='Boxes in Goal')

        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Cumulative Reward', color='g')
        ax2.set_ylabel('Number of Steps', color='b')
        ax3.set_ylabel('Boxes in Goal', color='r')

        ax1.tick_params(axis='y', labelcolor='g')
        ax2.tick_params(axis='y', labelcolor='b')
        ax3.tick_params(axis='y', labelcolor='r')

        ax3.set_ylim(0, 5)  # Set the y-axis limits for boxes in goal

        fig.tight_layout()
        fig.legend(loc='upper left', bbox_to_anchor=(0, 1), bbox_transform=ax1.transAxes)
        num_boxes_equals_5 = sum(1 for boxes in stats['boxes_in_goal'] if boxes == 5)
        print(f"Number of times boxes in goal equals 5: {num_boxes_equals_5}, out of {len(stats['boxes_in_goal'])} episodes")
        plt.show()
    

def main(args):
    test = args.test
    with open(args.config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    global env
    if config['log_file'] is not None:
        if test:
            logging.basicConfig(filename=config['log_file'][:-4]+'_test.log',level=logging.DEBUG)
        else:
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

    train = Train_DQL(config, test)
    
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
        default= 'configurations/config_basic_test.yml'
        # default= 'configurations/config_basic_primitive.yml'
    )

    parser.add_argument(
        '--test',
        type=bool,
        help='testing mode',
        default= False
    )

    main(parser.parse_args())