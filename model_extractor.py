import torch
from collections import namedtuple

options = False

checkpoint_path = 'checkpoint/checkpoint-basic-states-boxes.pt'
model_path = 'model_weights/model-basic-states-boxes.pt'
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'distance'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(checkpoint_path, map_location=device)
if options:
    for i in range(2):
        torch.save(checkpoint[f'policy_state_dict_{i}'], model_path[:-3] + f'_{i}.pt')
else:
    model = {
        'policy_state_dict_0': checkpoint['policy_state_dict_0'],
        'stats': checkpoint['stats'],
    }
    # torch.save(checkpoint['policy_state_dict_0'], model_path)
    torch.save(model, model_path)
# print(checkpoint['epsilon'])