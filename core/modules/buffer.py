'''
Online batch sampler
'''

import numpy as np

import torch
from torch.utils.data import Sampler

from collections import deque

class Buffer(Sampler):

    def __init__(self, T, size=8, cycles=1, batch_size=8, slide=True):
        self.T = T

        if size == -1:
            self.size = T
        else:
            self.size = size

        self.buffer = deque(maxlen=self.size)

        self.cycles = cycles

        self.batch_size = batch_size

        #wait until buffer is full before yielding
        self.slide = slide

    def __iter__(self):
        #iterate over snapshots
        for i in range(len(self)):

            #update buffer
            if i < self.T:
                self.buffer.append(i)
            elif self.slide:
                self.buffer.popleft()

            if not self.slide and i < self.size-1:
                continue

            torch_buffer = torch.tensor(list(self.buffer))

            #iterate over cycles
            for j in range(self.cycles):
                
                #shuffle
                split = torch.split(torch.randperm(len(self.buffer)), self.batch_size)

                for idxs in split:
                   yield torch_buffer[idxs]

    def __len__(self):
        return self.T + self.size - 1 if self.slide else self.T - 1