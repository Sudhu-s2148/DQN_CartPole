import math
import torch
import torch.nn as nn

def bellmans_update(active_network,dormant_network,buffer_exp,gamma):

    states_batch = torch.tensor([t[0] for t in buffer_exp], dtype=torch.float32)
    actions_batch = torch.tensor([t[1] for t in buffer_exp], dtype=torch.int64)
    next_states_batch = torch.tensor([t[2] for t in buffer_exp], dtype=torch.float32)
    rewards_batch = torch.tensor([t[3] for t in buffer_exp], dtype=torch.float32)
    dones_batch = torch.tensor([t[4] for t in buffer_exp], dtype=torch.float32)

    # target — computed from target network, no gradients needed
    with torch.no_grad():
        max_Q = dormant_network.forward(next_states_batch).max(dim=1).values
        max_Q = max_Q * (1 - dones_batch)
        target = rewards_batch + gamma * max_Q

    # prediction — computed from online network, gradients ARE needed
    predicted = active_network.forward(states_batch).gather(1, actions_batch.unsqueeze(1)).squeeze(1)

    # loss — PyTorch tracks gradients through predicted
    loss = nn.MSELoss()(predicted, target)
    return loss

def compute_reward(state_tensor):
    theta_normalized = state_tensor[2].item() % (2 * math.pi)
    upright = abs(theta_normalized - math.pi) < 0.5  # within ~28° of upright
    fallen = abs(theta_normalized - math.pi) > 0.5
    if upright:
        reward = 1
    elif state_tensor[0].item() <= 50 or state_tensor[0].item() >= 750:
        reward = -5
    else:
        reward = -1
    return reward, fallen


