import math
import torch
import torch.nn as nn


def bellmans_update(active_network,dormant_network,buffer_exp,gamma,device):

    states_batch = torch.tensor([t[0] for t in buffer_exp], dtype=torch.float32).to(device)
    actions_batch = torch.tensor([t[1] for t in buffer_exp], dtype=torch.int64).to(device)
    next_states_batch = torch.tensor([t[2] for t in buffer_exp], dtype=torch.float32).to(device)
    rewards_batch = torch.tensor([t[3] for t in buffer_exp], dtype=torch.float32).to(device)
    dones_batch = torch.tensor([t[4] for t in buffer_exp], dtype=torch.float32).to(device)

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
    upright = abs(theta_normalized - math.pi) < 0.8

    out_of_bounds = state_tensor[0].item() < 150 or state_tensor[0].item() > 650
    fallen = not upright or out_of_bounds

    angle_reward = 1 - abs(theta_normalized - math.pi) / math.pi
    position_reward = 1 - abs(state_tensor[0].item() - 400) / 400
    wall_distance = abs(state_tensor[0].item() - 400) / 400
    wall_penalty = wall_distance ** 2

    if upright:
        stillness_bonus = 1 / (1 + abs(state_tensor[1].item()) * 0.1)
        velocity_penalty = (abs(state_tensor[1].item()) / 60) ** 2 * 2
        rew = 2*angle_reward + position_reward + stillness_bonus - velocity_penalty - 4*wall_penalty
    else:
        velocity_penalty = (abs(state_tensor[1].item()) / 60) ** 2
        rew = 2*angle_reward + position_reward - velocity_penalty - 4*wall_penalty

    if out_of_bounds:
        rew = -20

    return rew, fallen


