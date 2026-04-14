from collections import deque
import random
class Buffer:
    def __init__(self):
        self.deq = deque(maxlen = 100000)
    def __len__(self):
        return len(self.deq)
    def push(self, state, action, next_state, reward, done):
        ele = (state, action, next_state, reward, done)
        self.deq.append(ele)
    def sample(self, batch_size):
        batch = random.sample(self.deq,batch_size)
        return batch