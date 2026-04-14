import numpy as np
import agent as A
import buffer
import torch
import torch.nn as nn
import gymnasium
import os

training = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

env = gymnasium.make("CartPole-v1")
online_network = A.agent().to(device)
offline_network = A.agent().to(device)
experiences = buffer.Buffer()
gamma = 0.99
epsilon = 1
epsilon_min = .05
decay = .9995
episode = 5000
step = 500

optimizer = torch.optim.Adam(online_network.parameters(), lr=0.001)

def bellmans_update(active_network,dormant_network,buffer_exp,gamma, device):

    states_batch = torch.tensor(np.array([t[0] for t in buffer_exp]), dtype=torch.float32).to(device)
    actions_batch = torch.tensor([t[1] for t in buffer_exp], dtype=torch.int64).to(device)
    next_states_batch = torch.tensor(np.array([t[2] for t in buffer_exp]), dtype=torch.float32).to(device)
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

total_overall_steps = 0
total_steps = 0
best_avg_reward = 0
#episode loop
if training:
    print("started")
    for j in range(episode):
        save_path = "best_agent.pth"
        step_count = 0
        state,_ = env.reset()
        state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
        for i in range(step):
            total_overall_steps+=1
            step_count+=1
            action = online_network.choice(state_tensor,epsilon)# forward pass
            next_state, reward, terminated, truncated, _ = env.step(action)
            experiences.push(state,action,next_state,reward,terminated)

            state = next_state
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
            if len(experiences)>1000:
                batch = experiences.sample(64)
                optimizer.zero_grad()  # reset gradients
                loss = bellmans_update(online_network,offline_network,batch,gamma,device) # compute loss
                loss.backward() # compute gradients
                optimizer.step()# update weights

            if total_overall_steps % 1000 == 0:
                offline_network.load_state_dict(online_network.state_dict())
            if terminated or truncated:
                break
        total_steps += (i + 1)

        if (j + 1) % 100 == 0:
            current_avg = total_steps / 100

            if current_avg >= best_avg_reward:
                best_avg_reward = current_avg
                torch.save(online_network.state_dict(), save_path)
                print(f"--> New Best Model Saved! Avg Steps: {best_avg_reward:.1f}")
            print(f"Episode: {j+1} | Avg Steps (last 100): {total_steps/100:.1f} | Epsilon: {epsilon:.3f}")
            total_steps = 0

        epsilon = max(epsilon * decay, epsilon_min)
    env.close()
else:
    model = A.agent()
    model.to(device)
    if os.path.exists("best_cartpole_model.pth"):
        model.load_state_dict(torch.load("best_cartpole_model.pth", map_location=device))
        model.eval()
        print("Model loaded successfully!")
    else:
        print("File not found, using fresh model.")
    env = gymnasium.make("CartPole-v1", render_mode="human")
    state, _ = env.reset()
    done = False
    while not done:
        action = model.choice(torch.tensor(state, dtype=torch.float32).to(device), epsilon=0)  # epsilon=0 means no randomness
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    env.close()
"""theta_normalized = state_tensor[2].item() % (2 * math.pi)
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

        if i == 499 or fallen: done = True"""