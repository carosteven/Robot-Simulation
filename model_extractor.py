import torch
from collections import namedtuple

options = True

checkpoint_path = 'checkpoint/checkpoint-options_rews.pt'
model_path = 'model_weights/model_options_rews.pt'
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'distance'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(checkpoint_path, map_location=device)
if options:
    for i in range(2):
        torch.save(checkpoint[f'policy_state_dict_{i}'], model_path[:-3] + f'_{i}.pt')
else:
    torch.save(checkpoint['policy_state_dict'], model_path)
# print(checkpoint['epsilon'])