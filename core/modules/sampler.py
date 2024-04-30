'''
Online batch sampler
'''

import random

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

        self.length = self.T + self.size - 1 if self.slide else self.T - 1

    def __iter__(self):
        #iterate over snapshots
        for i in range(self.length):

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
        return self.length*self.cycles
    
class Window(Sampler):

    def __init__(self, T, window_size=8, cycles=1):

        self.T = T
        self.window_size = window_size
        self.cycles = cycles

    def __iter__(self):

        for i in range(self.T//self.window_size):
            for j in range(self.cycles):
                yield list(range(i*self.window_size, (i+1)*self.window_size))

    def __len__(self):
        return (self.T//self.window_size)*self.cycles
    
class Queue(Sampler):

    def __init__(self, T, step=8, cycles=1, batch_size=8):

        self.T = T
        self.step = step
        self.cycles = cycles
        self.batch_size = batch_size
        self.queue = []

    def __iter__(self):
        for i in range(self.T//self.step):
            for j in range(self.cycles):
                if len(self.queue) <= self.batch_size:
                    yield self.queue
                else:
                    yield random.sample(self.queue, k=self.batch_size)
            
            self.queue.extend(list(range(i*self.step, (i+1)*self.step)))

    def __len__(self):
        return (self.T//self.step)*self.cycles