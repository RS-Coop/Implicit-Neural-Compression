'''
Online batch sampler
'''

import random

from torch.utils.data import Sampler

from collections import deque

class Buffer(Sampler):

    def __init__(self, T, size=1, cycles=1, batch_size=1, step=1, delay=False):
        #Hyperparameters
        self.T = T

        if size == -1:
            self.size = T
        else:
            self.size = size

        self.cycles = cycles
        self.batch_size = batch_size
        self.step = step
        self.delay = delay

        assert (0<self.size and self.size<=self.T), f"Buffer size ({self.size}) must be positive and no larger than time horizon ({self.T})"
        assert (0<self.batch_size and self.batch_size<=self.size), f"Batch size ({self.batch_size}) must be positive and no larger than buffer size ({self.size})"
        assert (0<self.step and self.step<=self.T+self.size), f"Step ({self.step}) must be positive and no larger than ({self.T+self.size})"

        #Buffer
        self.buffer = deque(maxlen=self.size)

        #Length
        self.length = self.T//self.step + self.T%self.size

        return

    def __iter__(self):
        delay = self.delay

        #iterate over snapshots
        #NOTE: I wrote the following loop but am a bit confused by it
        for i, _ in zip(range(0, self.T+self.size, self.step), range(self.length)):

            if delay:
                for j in range(self.cycles): yield list(self.buffer)
                delay = False

            #update buffer
            for idx in range(i,i+self.step):
                if idx < self.T:
                    self.buffer.append(idx)
                else:
                    self.buffer.popleft()

            #iterate over cycles
            for j in range(self.cycles):
                if len(self.buffer) <= self.batch_size:
                    yield list(self.buffer)
                else:
                    yield random.sample(list(self.buffer), k=self.batch_size)

        return

    def __len__(self):
        return self.length*self.cycles