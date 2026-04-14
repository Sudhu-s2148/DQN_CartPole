from collections import deque
import random
class Buffer:
    def __init__(self):
        self.deq = deque(maxlen = 50000)
    def __len__(self):
        return len(self.deq)
    def push(self, state, action, next_state, reward, done):
        ele = (state, action, next_state, reward, done)
        self.deq.append(ele)
    def sample(self, batch_size):
        buffer_list = list(self.deq)
        # need at least 1000 recent + 51 old to use priority sampling
        if len(buffer_list) < 1051:
            return random.sample(buffer_list, batch_size)

        recent = buffer_list[-1000:]
        old = buffer_list[:-1000]

        n_recent = int(batch_size * 0.8)
        n_old = batch_size - n_recent
        return random.sample(recent, n_recent) + random.sample(old, n_old)