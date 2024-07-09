import torch

checkpoint_path = 'checkpoint/checkpoint.pt'
model_path = 'model_push.pt'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(checkpoint_path, map_location=device)
torch.save(checkpoint['policy_state_dict'], model_path)