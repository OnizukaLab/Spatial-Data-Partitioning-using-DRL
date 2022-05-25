import random
import numpy as np

class ReplayMemory(object):

    def __init__(self, capaity):
        self.capacity = capaity
        self.memory = [] 
        self.index = 0 

    def push(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None) 

        self.memory[self.index] = transition
        self.index = (self.index + 1) % self.capacity 

    def sample(self, batch_size, episode=0, num_episodes=0):
        return None, random.sample(self.memory, batch_size), np.ones((batch_size))

    def __len__(self):
        return len(self.memory)
        

# Prioritized Experience Reply Memory
class PERMemory(ReplayMemory):
    EPSILON = 0.0001
    ALPHA = 0.6
    BETA = 0.4
    size = 0

    def __init__(self, capacity):
        super(PERMemory, self).__init__(capacity)
        self.capacity = capacity
        self.tree = SumTree(capacity)

    def _getPriority(self, td_error):
        return (td_error + self.EPSILON) ** self.ALPHA

    def push(self, transition):
        self.size += 1

        priority = self.tree.max()
        if priority <= 0:
            priority = 1

        self.tree.add(priority, transition)

    def sample(self, size, episode, num_episodes):
        list = []
        indexes = []
        weights = np.empty(size, dtype='float32')
        total = self.tree.total()
        beta = self.BETA + (1 - self.BETA) * episode / num_episodes

        for i, rand in enumerate(np.random.uniform(0, total, size)):
            (idx, priority, data) = self.tree.get(rand)
            list.append(data)
            indexes.append(idx)
            weights[i] = (self.capacity * priority / total) ** (-beta)

        return (indexes, list, weights / weights.max())

    def update(self, idx, td_error):
        priority = self._getPriority(td_error)
        self.tree.update(idx, priority)

    def __len__(self):
        return self.size


class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2*capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.index_leaf_start = capacity - 1

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def max(self):
        return self.tree[self.index_leaf_start:].max()

    def add(self, p, data):
        idx = self.write + self.index_leaf_start

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])