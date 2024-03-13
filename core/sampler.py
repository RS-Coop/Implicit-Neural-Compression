'''
Online batch sampler
'''

from torch.utils.data import Sampler

from collections import deque

class OnlineSampler(Sampler):

    def __init__(self, data_source, buffer_size=4):
        self.data_source = data_source

        self.buffer = deque(buffer_size)

    def __iter__(self):
        N = self.data_source.num_points
        T = self.data_source.num_snapshots

        for i in range(len(self)):
            if i < T:
                self.buffer.append(range(i*N,(i+1)*N))
            else:
                self.buffer.pop()

            yield list(deque)

    def __len__(self):
        T = self.data_source.num_snapshots

        return T+self.buffer_size-1
    
class WindowSampler(Sampler):

    def __init__(self, N, T, window_size=4):

        self.N = N
        self.T = T
        self.window_size = window_size

    def __iter__(self):
        
        return iter([range(i*self.N,(i+self.window_size)*self.N) for i in range(len(self))])

    def __len__(self):

        return self.T-self.window_size+1