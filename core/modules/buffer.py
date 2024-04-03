'''
Online batch sampler
'''

import numpy as np

import torch
from torch.utils.data import Sampler

from collections import deque

class Buffer(Sampler):

    def __init__(self, T, buffer_size=8, cycles=1, batch_size=4):
        self.T = T

        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

        self.cycles = cycles

        self.batch_size = batch_size

        self.num_batches = int(np.ceil(buffer_size/batch_size))

    def __iter__(self):

        #iterate over snapshots
        for i in range(len(self)):

            #update buffer
            if i < self.T:
                self.buffer.append(i)
            else:
                self.buffer.popleft()

            torch_buffer = torch.tensor(list(self.buffer))

            #iterate over cycles
            for j in range(self.cycles):
                
                #shuffle
                s = torch.randperm(len(self.buffer))

                for batch_idxs in torch.split(s, self.batch_size):
                    yield torch_buffer[batch_idxs]

    def __len__(self):

        return self.T+self.buffer_size-1