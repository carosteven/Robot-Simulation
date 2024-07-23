import torch
from collections import namedtuple

checkpoint_path = 'checkpoint/checkpoint_pascal.pt'
model_path = 'model_push.pt'
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(checkpoint_path, map_location=device)
torch.save(checkpoint['policy_state_dict'], model_path)
print(checkpoint['epsilon'])