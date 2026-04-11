import math
import agent as A
import buffer
import statics5
import torch
import torch.nn as nn
online_network = A.agent()
offline_network = A.agent()
experiences = buffer.Buffer()
gamma = 0.99

#reward system

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

optimizer = torch.optim.Adam(online_network.parameters(), lr=0.001)
epsilon = 1

state = statics5.state_before_action
state_tensor = torch.tensor(state, dtype=torch.float32)

#episode loop
for j in range(10):
    #step loop
    done = False
    for i in range(500):

        action = online_network.choice(state_tensor,epsilon)  # forward pass
        next_state = statics5.state_after_action
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32)

        theta_normalized = state_tensor[2].item() % (2 * math.pi)
        fallen = theta_normalized > 30 * math.pi / 180 and theta_normalized < (2 * math.pi - 30 * math.pi / 180)
        upright = theta_normalized < 20 * math.pi / 180 or theta_normalized > (2 * math.pi - 20 * math.pi / 180)
        if upright:
            reward = 1
        elif state_tensor[0].item() <= 50 or state_tensor[0].item() >= 750:
            reward = -5
        elif not upright and not fallen:
            reward = 0
        else:
            reward = -1

        if i == 499 or fallen: done = True

        experiences.push(state,action,next_state,reward,done)
        optimizer.zero_grad()            # reset gradients
        if i>64:
            batch = experiences.sample(64)
            loss = bellmans_update(online_network,offline_network,batch,gamma) # compute loss
            loss.backward() # compute gradients
            optimizer.step()# update weights
        state = next_state
        state_tensor = next_state_tensor
        if done: break
    offline_network.load_state_dict(online_network.state_dict())
    epsilon *= .99
