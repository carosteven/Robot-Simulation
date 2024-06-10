# Adapted from https://github.com/jimmyyhwu/spatial-action-maps/blob/master/policies.py
import random
import torch
from torchvision import transforms
import resnet #models
import numpy as np

class DQNPolicy:
    def __init__(self, cfg, action_space, train=False, random_seed=None):
        self.cfg = cfg
        self.action_space = action_space
        self.train = train

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net = self.build_network()
        self.transform = transforms.ToTensor()
        # self.rgb_conv = torch.from_numpy((np.ones((3,600,600))*255).astype('int'))
        self.rgb_conv = (np.ones((200,200,3))*255).astype('int')

        # Resume from checkpoint if applicable
        '''if self.cfg.checkpoint_path is not None:
            model_checkpoint = torch.load(self.cfg.model_path, map_location=self.device)
            self.policy_net.load_state_dict(model_checkpoint['state_dict'])
            if self.train:
                self.policy_net.train()
                
            else:
                self.policy_net.eval()
            print("=> loaded model '{}'".format(self.cfg.model_path))'''

        if random_seed is not None:
            random.seed(random_seed)

    def build_network(self):
        raise NotImplementedError

    def apply_transform(self, s):
        # print()
        # print(s[300,300])
        # s = torch.from_numpy(s)
        # t = torch.from_numpy(np.zeros((1,3,600,600)))
        '''
        t[0,0] = torch.bitwise_right_shift(s,16)
        t[0,1] = torch.bitwise_right_shift(s,8)
        t[0,2] = s'''
        # t = torch.bitwise_and(torch.transpose(torch.from_numpy(np.array([np.right_shift(s,16), np.right_shift(s,8), s])).to(self.device),1,2), self.rgb_conv)
        t = np.bitwise_and(np.transpose(np.array([np.right_shift(s,16), np.right_shift(s,8), s])), self.rgb_conv)
        # t[0] = torch.bitwise_and(t[0], self.rgb_conv)
        # print(t.shape)
        return self.transform(t).unsqueeze(0).float()
        # return t.unsqueeze(0).float()

    def step(self, state, exploration_eps=None, debug=False):
        if exploration_eps is None:
            exploration_eps = self.cfg['final_exploration']
        state = self.apply_transform(state).to(self.device)
        with torch.no_grad():
            output = self.policy_net(state).squeeze(0)
        if random.random() < exploration_eps:
            action = random.randrange(self.action_space)
        else:
            action = output.view(1, -1).max(1)[1].item()
        info = {}
        if debug:
            info['output'] = output.squeeze(0)
        return action, info

class SteeringCommandsPolicy(DQNPolicy):
    def build_network(self):
        return torch.nn.DataParallel(
            resnet.SteeringCommandsDQN(num_input_channels=self.cfg['num_input_channels'], num_output_channels=self.action_space)
        ).to(self.device)