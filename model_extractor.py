import torch
from collections import namedtuple

checkpoint_path = 'checkpoint/checkpoint-push_rews.pt'
model_path = 'model_weights/model_push_rews.pt'
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'distance'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(checkpoint_path, map_location=device)
torch.save(checkpoint['policy_state_dict'], model_path)
print(checkpoint['epsilon'])