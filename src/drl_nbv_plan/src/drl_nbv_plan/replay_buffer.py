#!/usr/bin/env python3
import random
import collections
from tree import SumTree 
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = collections.deque(maxlen=buffer_size)
        self.rng = np.random.default_rng()  # Faster RNG

    def add(self, experience:list):
        self.buffer.append(experience)
        if len(self.buffer) > self.buffer_size:
            self.buffer.popleft()
            print("buffer is full, clearing buffer")

    def sample_batch(self, size):
        # Faster sampling using numpy
        indices = self.rng.choice(len(self.buffer), size=size, replace=False)
        return [self.buffer[i] for i in indices]

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

class PrioritizedReplayBuffer:
    def __init__(self, buffer_size, eps=1e-2, alpha=0.1, beta=0.1):
        self.buffer = collections.deque(maxlen=buffer_size)
        self.tree = SumTree(size=buffer_size)

        # PER params
        self.eps = eps  # minimal priority, prevents zero probabilities
        self.alpha = alpha  # determines how much prioritization is used, α = 0 corresponding to the uniform case
        self.beta = beta  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
        self.max_priority = eps  # priority for new samples, init as eps

        self.buffer_size = buffer_size
        self.count = 0
        self.buffer_full = False

    #def add(self, experience:list):

        #if not self.buffer_full:
            #self.tree.add(self.max_priority, self.count)
            #self.buffer.append(experience)
            #self.count += 1
            #if self.count == self.buffer_size:
               # self.buffer_full = True
        #else:
            #self.count = self.count % self.buffer_size
            #self.tree.add(self.max_priority, self.count)
            #buffer = np.array(self.buffer)
            #buffer[self.count] = experience
            #self.buffer = collections.deque(buffer, maxlen=self.buffer_size)
            #self.count += 1
            
    def add(self, experience:list):
            if not self.buffer_full:
                self.buffer.append(experience)
            else:
                self.buffer[self.count] = experience

            self.tree.add(self.max_priority, self.count)
            self.count = (self.count + 1) % self.buffer_size
            if self.count == 0:
                self.buffer_full = True
            
    def sample_batch(self, batch_size):
        sample_idxs, tree_idxs = [], []
        priorities = torch.empty(batch_size, 1, dtype=torch.float32)
        segment = self.tree.total / batch_size

        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)

            cumsum = random.uniform(a, b)
            # sample_idx is a sample index in buffer, needed further to sample actual transitions
            # tree_idx is a index of a sample in the tree, needed further to update priorities
            result = self.tree.get(cumsum)
            if result is not None:
                tree_idx, priority, sample_idx = result
                priorities[i] = float(priority)
                tree_idxs.append(tree_idx)
                sample_idxs.append(sample_idx)
        
        # Concretely, we define the probability of sampling transition i as P(i) = p_i^α / \sum_{k} p_k^α
        # where p_i > 0 is the priority of transition i. (Section 3.3)
        probs = priorities / self.tree.total
        weights = (len(self.buffer) * probs) ** -self.beta
        weights = weights / weights.max()
        batch = [self.buffer[idx] for idx in sample_idxs]        
        return batch, tree_idxs, weights

    def update_priorities(self, data_idxs, priorities):
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.detach().cpu().numpy()

        for data_idx, priority in zip(data_idxs, priorities):
            # The first variant we consider is the direct, proportional prioritization where p_i = |δ_i| + eps,
            # where eps is a small positive constant that prevents the edge-case of transitions not being
            # revisited once their error is zero. (Section 3.3)
            priority = (priority + self.eps) ** self.alpha

            self.tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)
        
    def clear(self):
        self.buffer.clear()

    def __len__(self):
        pass

class RecurrentBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = collections.deque(maxlen=buffer_size)
    
    def add(self, experience:list):
        self.buffer.append(experience)
        if len(self.buffer) > self.buffer_size:
            self.buffer.popleft()

    def sample_batch(self, size):
        batch = random.sample(self.buffer, size)
        return batch
