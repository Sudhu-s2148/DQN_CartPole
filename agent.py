import torch
import random
import torch.nn as nn
import torch.nn.functional as F
class agent(nn.Module):
  def __init__(self):
    super(agent,self).__init__()
    self.input = nn.Linear(4,64)
    self.layer1 = nn.Linear(64,64)
    self.layer2 = nn.Linear(64, 64)
    self.output = nn.Linear(64,2)
  def forward(self,x):
    x = F.relu(self.input(x))
    x = F.relu(self.layer1(x))
    x = F.relu(self.layer2(x))
    x = self.output(x)
    return x
  def choice(self,state,epsilon):
    if random.random()>epsilon:
      with torch.no_grad():
        return torch.argmax(self.forward(state))
    else:
      return random.randint(0,1)
