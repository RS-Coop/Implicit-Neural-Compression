'''
Online batch sampler
'''

from torch.utils.data import Sampler

from collections import deque

class OnlineSampler(Sampler):

    def __init__(self, N, T, buffer_size=3):
        self.N = N
        self.T = T

        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def __iter__(self):

        for i in range(len(self)):
            if i < self.T:
                self.buffer.append(i)
            else:
                self.buffer.popleft()

            yield range(self.buffer[0]*self.N,(self.buffer[-1]+1)*self.N)

    def __len__(self):

        return self.T+self.buffer_size-1
    
class WindowSampler(Sampler):

    def __init__(self, N, T, window_size=4):

        self.N = N
        self.T = T
        self.window_size = window_size

    def __iter__(self):
        
        return iter([range(i*self.N,(i+self.window_size)*self.N) for i in range(len(self))])

    def __len__(self):

        return self.T-self.window_size+1