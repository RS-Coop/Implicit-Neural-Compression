'''
Online batch sampler
'''

import numpy as np

import torch
from torch.utils.data import Sampler

from collections import deque

class Buffer(Sampler):

    def __init__(self, T, hifi_size=2, lofi_size=-1, cycles=1, hifi_batch_size=2, lofi_batch_size=5):
        self.T = T

        if lofi_size==-1: lofi_size = T

        self.hifi_size = hifi_size
        self.lofi_size = lofi_size

        self.hifi_buffer = deque(maxlen=hifi_size)
        self.lofi_buffer = deque(maxlen=lofi_size)

        self.cycles = cycles

        self.hifi_batch_size = hifi_batch_size
        self.lofi_batch_size = lofi_batch_size

        self.length = T + hifi_size - 1

    def __iter__(self):
        #iterate over snapshots
        for i in range(len(self)):

            #update buffer
            if i < self.T:
                self.hifi_buffer.append(i)
                self.lofi_buffer.append(i)
            else:
                self.hifi_buffer.popleft()

            hifi_torch_buffer = torch.tensor(list(self.hifi_buffer))
            lofi_torch_buffer = torch.tensor(list(self.lofi_buffer))

            #iterate over cycles
            for j in range(self.cycles):
                
                #shuffle
                hifi_split = torch.split(torch.randperm(len(self.hifi_buffer)), self.hifi_batch_size)
                lofi_split = torch.split(torch.randperm(len(self.lofi_buffer)), self.lofi_batch_size)

                for hifi_idxs, lofi_idxs in zip(hifi_split, lofi_split):
                   yield (hifi_torch_buffer[hifi_idxs], lofi_torch_buffer[lofi_idxs])

    def __len__(self):
        return self.length