import numpy as np
import torch
from .. import partition as build_partition

class Rosenbrock(torch.nn.Module):
    def __init__(self, d, a, b):
        super(Rosenbrock, self).__init__()

        self.d = d
        self.a = a
        self.b = b

        self.xeven = torch.nn.Parameter(torch.randn(d))
        self.xodd = torch.nn.Parameter(torch.randn(d))

    def forward(self, x):
        return ((self.xodd - self.a).pow(2) + self.b * (self.xodd.pow(2) - self.xeven).pow(2)).sum()

class RosenbrockT(torch.nn.Module):
    def __init__(self, d, a, b):
        super(RosenbrockT, self).__init__()

        self.d = d
        self.a = a
        self.b = b

        for i in range(d):
            self.register_parameter('mod{:03}'.format(i), torch.nn.Parameter(torch.randn(2)))

    def forward(self, x):
        result = 0
        for i in range(self.d):
            m = getattr(self, 'mod{:03}'.format(i))
            result = result + (m[1] - self.a).pow(2) + self.b * (m[1].pow(2) - m[0]).pow(2)

        return result
