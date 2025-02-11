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
from torchvision import transforms
from tqdm import tqdm

import os
import sys
import yaml

import logging

import environments
import models

from PIL import Image

logging.getLogger('pymunk').propagate = False

env = None

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'ministep'))



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
    def __init__(self, config, job_id, test):
        print(config['job_name'])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_type = config['policy_type']
        self.checkpoint_path = f'checkpoint/{job_id}/checkpoint-{config["job_name"]}.pt' if not test else f'model_weights/model-{config["job_name"]}.pt'
        self.checkpoint_freq = config['checkpoint_freq']
        self.total_timesteps = config['total_timesteps']
        self.learning_starts = config['learning_starts']
        self.grad_norm_clipping = config['grad_norm_clipping']
        self.optimizer_type = config['optimizer_type']
        self.test = test
        self.resume_training = config['resume_training']
        self.job_id_to_resume = config['job_id_to_resume']
        self.job_name = config['job_name']
        self.num_input_channels = config['num_input_channels']
        # Get number of actions from env
        self.action_space = env.get_action_space()

        self.checkpoint_counter = 0
        self.total_checkpoints = int(12*60*60 / self.checkpoint_freq) # 12 hours

        # Global variables
        self.BATCH_SIZE = config['batch_size']                  # How many examples to sample per train step
        self.GAMMA = config['discount_factor']                  # Discount factor in episodic reward objective
        self.LEARNING_RATE = config['learning_rate']            # Learning rate for optimizer
        self.WEIGHT_DECAY = config['weight_decay']              # Weight decay for optimizer
        self.TARGET_UPDATE_FREQ = config['target_update_freq']  # Target network update frequency
        self.STARTING_EPSILON = config['starting_exploration']  # Starting epsilon
        self.STEPS_MAX = config['exploration_timesteps']        # Gradually reduce epsilon over these many steps
        self.EPSILON_END = config['final_exploration']          # At the end, keep epsilon at this value

        self.EPSILON = self.STARTING_EPSILON

        self.state = self.get_state(env.reset())
        
        self.episodic_stats = {'cumulative_reward': [], 'num_steps': [], 'boxes_in_goal': []}
        
        self.policy = self.create_or_restore_training_state(config['model'], config['replay_buffer_size'])

        self.steps_done = 0 # for exploration
        self.last_epi_box_in_goal = 0

    
    def get_state(self, raw_state):
        state = raw_state
        return state


    def create_or_restore_training_state(self, model, buffer_size):
        if model == 'resnet':
            if self.policy_type == 'dense_action_space':
                policy_net = models.VisionDQN_SAM(self.num_input_channels)
                target_net = models.VisionDQN_SAM(self.num_input_channels)
            elif self.policy_type == 'steering_commands':
                policy_net = models.VisionDQN(self.num_input_channels, self.action_space)
                target_net = models.VisionDQN(self.num_input_channels, self.action_space)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        if self.test:
            policy_net.eval()
        else:
            policy_net.train()

        # self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.LEARNING_RATE)
        if self.optimizer_type == 'sgd':
            optimizer = optim.SGD(policy_net.parameters(), lr=self.LEARNING_RATE, momentum=0.9, weight_decay=self.WEIGHT_DECAY)
        elif self.optimizer_type == 'adamw':
            optimizer = optim.AdamW(policy_net.parameters(), lr=self.LEARNING_RATE, weight_decay=0.01)
        memory = ReplayMemory(buffer_size)
        self.epoch = 0
        loss = 0
        epsilon = self.STARTING_EPSILON
        if os.path.exists(self.checkpoint_path) or self.resume_training:
            if self.resume_training:
                self.resume_training = False # Only true at the start (use current checkpoint path if preemption)
                old_checkpoint_path = f'checkpoint/{self.job_id_to_resume}/checkpoint-{self.job_name}.pt'
                training_state = torch.load(old_checkpoint_path, map_location=self.device)
            else:
                training_state = torch.load(self.checkpoint_path, map_location=self.device)

            # Remove the last element from the stats since it is not complete
            for key in training_state['stats'].keys():
                if len(training_state['stats'][key]) > 0:
                    training_state['stats'][key].pop()
            self.episodic_stats = training_state['stats']

            if self.test:
                policy_net.load_state_dict(training_state[f'policy_state_dict'])
                policy_net.eval()
                self.show_stats(self.episodic_stats)
                epsilon = self.EPSILON_END

            else:
                # training_state = torch.load(self.checkpoint_path)
                self.epoch = training_state['epoch']
                policy_net.load_state_dict(training_state[f'policy_state_dict'])
                target_net.load_state_dict(training_state[f'target_state_dict'])
                optimizer.load_state_dict(training_state[f'optimizer_state_dict'])
                memory.memory = training_state[f'memory']
                loss = training_state[f'loss']
                epsilon = training_state[f'epsilon']
                logging.info(f"Training state restored at epoch {self.epoch}")
        else:
            logging.info("No checkpoint detected, starting from initial state")

        return {'policy_net': policy_net, 'target_net': target_net, 'optimizer': optimizer, 'memory': memory, 'loss': loss, 'epsilon': epsilon}

    def commit_state(self):
        temp_path = os.path.join(os.path.dirname(self.checkpoint_path), "temp.pt")
        training_state = {}
        training_state['epoch'] = self.epoch
        training_state['stats'] = self.episodic_stats
        training_state[f'policy_state_dict'] = self.policy['policy_net'].state_dict()
        training_state[f'target_state_dict'] = self.policy['target_net'].state_dict()
        training_state[f'optimizer_state_dict'] = self.policy['optimizer'].state_dict()
        training_state[f'memory'] = self.policy['memory'].memory
        training_state[f'loss'] = self.policy['loss']
        training_state[f'epsilon'] = self.policy['epsilon']

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
    
    def transform(self, state):
        return transforms.ToTensor()(state).unsqueeze(0)
    
    def get_action(self, policy):
        # With probability EPSILON, choose a random action
        # Rest of the time, choose argmax_a Q(s, a) 
        if np.random.rand() < policy['epsilon']:
            action = np.random.randint(self.action_space)

        else:
            with torch.no_grad():
                state = self.transform(self.state).to(self.device)
                qvalues = policy['policy_net'](state)
            action = torch.argmax(qvalues).item()
        # action = torch.tensor(action, device=self.device, dtype=torch.long)
        
        # Epsilon update rule: Keep reducing a small amount over
        # STEPS_MAX number of steps, and at the end, fix to EPSILON_END
        prev_eps = policy['epsilon']
        policy['epsilon'] = max(self.EPSILON_END, policy['epsilon'] - (1.0 / self.STEPS_MAX))
        if policy['epsilon'] == self.EPSILON_END and policy['epsilon'] != prev_eps:
            logging.info("Reached min epsilon")

        return action
    
    def update_networks(self, policy, epi):
        # Sample a minibatch (s, a, r, s', d)
        # Each variable is a vector of corresponding values
        transitions = policy['memory'].sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([self.transform(s) for s in batch.next_state if s is not None]).to(self.device)
        state_batch = torch.cat([self.transform(s) for s in batch.state]).to(self.device)
        action_batch = torch.tensor(batch.action, dtype=torch.long).to(self.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(self.device)
        ministep_batch = torch.tensor(batch.ministep, device=self.device, dtype=torch.float)

        # Get Q(s, a) for every (s, a) in the minibatch
        output = policy['policy_net'](state_batch)
        qvalues = output.view(self.BATCH_SIZE, -1).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        
        # Double DQN Formula: r + gamma*TARGET(s_t+1, argmax_a POLICY (s_t+1, a))
        q_target_values = torch.zeros(self.BATCH_SIZE, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            # best_actions = torch.argmax(policy['policy_net'](non_final_next_states), dim=1)
            # q_target_values[non_final_mask] = policy['target_net'](non_final_next_states).gather(1, best_actions.unsqueeze(1)).squeeze()
            best_actions = policy['policy_net'](non_final_next_states).view(non_final_next_states.size(0), -1).max(1)[1].view(non_final_next_states.size(0), 1)
            q_target_values[non_final_mask] = policy['target_net'](non_final_next_states).view(non_final_next_states.size(0), -1).gather(1, best_actions).view(-1)
        targets = reward_batch + torch.pow(self.GAMMA, ministep_batch) * q_target_values

        # Detach y since it is the target. Target values should
        # be kept fixed.
        loss = F.smooth_l1_loss(targets.detach().view_as(qvalues), qvalues)

        # Backpropagation
        policy['optimizer'].zero_grad()
        loss.backward()
        if self.grad_norm_clipping is not None:
            torch.nn.utils.clip_grad_norm_(policy['policy_net'].parameters(), self.grad_norm_clipping)
        policy['optimizer'].step()

        # Update target network every few steps
        if epi % self.TARGET_UPDATE_FREQ == 0:
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
        self.policy['policy_net'] = self.policy['policy_net'].to(self.device)
        if not self.test:
            self.policy['target_net'] = self.policy['target_net'].to(self.device)
            self.policy['optimizer']  = self.optimizer_to_dev(self.policy['optimizer'])

        # for epoch in tqdm(range(self.num_epochs)):
        # for epoch in count():
        # Reset environment and get new state
        self.state = self.get_state(env.reset())
        # logging.info(f'Epoch {self.epoch}')

        # Keep track of stats
        self.episodic_stats['cumulative_reward'].append(0)
        self.episodic_stats['num_steps'].append(0)
        self.episodic_stats['boxes_in_goal'].append(0)

        # actions = []
        epi = 0
        self.last_epi_box_in_goal = 0
        done = False
        timeout = False
        total_timesteps_with_warmup = self.total_timesteps + self.learning_starts
        for timestep in tqdm(range(total_timesteps_with_warmup)):
            action = self.get_action(self.policy)
            next_state, reward, done, timeout, info = env.step(action)

            if not self.test:
                self.policy['memory'].push(self.state, action, next_state, reward, info['ministeps'])

                # Train after collecting sufficient experience
                if timestep > self.learning_starts:
                    self.update_networks(self.policy, epi)

            self.state = next_state # will be overwritten if done (next_state is None)

            # Update stats
            self.episodic_stats['cumulative_reward'][-1] += reward.item()
            self.episodic_stats['num_steps'][-1] = timestep + 1
            self.episodic_stats['boxes_in_goal'][-1] = info['cumulative_cubes']

            cur_time = time.time()
            if cur_time - start_time > self.checkpoint_freq and not self.test:
                self.commit_state()
                start_time = cur_time
                self.checkpoint_counter += 1
                if self.checkpoint_counter == self.total_checkpoints:
                    self.model_extractor()
                    logging.info("Time limit reached. Exiting training...")
                    sys.exit()
            
            if done:
                if timeout:
                    logging.info(f"Inactivity timeout. {info['cumulative_cubes']} in goal. Resetting environment...")
                else:
                    logging.info("All boxes in receptacle. Resetting environment...")
                self.state = self.get_state(env.reset())
                # logging.info(f'Epoch {self.epoch}')

                # Keep track of stats
                self.episodic_stats['cumulative_reward'].append(0)
                self.episodic_stats['num_steps'].append(0)
                self.episodic_stats['boxes_in_goal'].append(0)

                # actions = []
                epi += 1
                self.last_epi_box_in_goal = 0
                done = False
                timeout = False

        self.epoch += 1
    
    def model_extractor(self):
        model_path = f'model_weights/model-{env.config["job_name"]}.pt'
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        if self.options:
            for i in range(2):
                torch.save(checkpoint[f'policy_state_dict_{i}'], model_path[:-3] + f'_{i}.pt')
        else:
            model = {
                'policy_state_dict_0': checkpoint['policy_state_dict_0'],
                'stats': checkpoint['stats'],
            }
            torch.save(model, model_path)
    
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

        ax3.set_ylim(0, 10)  # Set the y-axis limits for boxes in goal

        fig.tight_layout()
        fig.legend(loc='upper left', bbox_to_anchor=(0, 1), bbox_transform=ax1.transAxes)
        num_boxes_equals_5 = sum(1 for boxes in stats['boxes_in_goal'] if boxes == 5)
        print(f"Number of times boxes in goal equals 5: {num_boxes_equals_5}, out of {len(stats['boxes_in_goal'])} episodes")
        plt.show()
    

def main(args):
    test = args.test
    job_id = args.job_id
    with open(args.config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    global env
    log_dir = f'output_logs/{config["job_name"]}.log'
    if test:
        logging.basicConfig(filename=log_dir[:-4]+'_test.log',level=logging.DEBUG)
    else:
        logging.basicConfig(filename=log_dir,level=logging.DEBUG)
    
    logging.info("starting training script")
    logging.info(f"Job ID: {job_id}")
    env = environments.selector(config)

    train = Train_DQL(config, job_id, test)
    
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
        # default= 'configurations/config_basic_eval.yml'
        # default= 'configurations/config_basic_test.yml'
        default= 'configurations/small_empty-steering_commands.yml'
        # default= 'configurations/config_cmplx_eval.yml'
        # default= 'configurations/config_cmplx_test.yml'
    )

    parser.add_argument(
        '--job_id',
        type=str,
        help='slurm job id',
        default= 'test'
    )

    parser.add_argument(
        '--test',
        type=bool,
        help='testing mode',
        default= False
    )

    main(parser.parse_args())