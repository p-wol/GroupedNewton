import numpy as np
import torch

class LeNet(torch.nn.Module):
    def __init__(self, layers, act_function, scaling = False, sigma_w = 1., sigma_b = 1., \
            sampler_w = lambda t: t.normal_(), sampler_b = lambda t: t.normal_(), first_layer_normal = False):
        super(LeNet, self).__init__()

        self.act_function = act_function
        self.scaling = scaling
        self.sigma_w = sigma_w
        self.sigma_b = sigma_b
        self.sampler_w = sampler_w
        self.sampler_b = sampler_b
        self.first_layer_normal = first_layer_normal

        self.conv1 = torch.nn.Conv2d(layers[0], layers[1], 5)
        self.conv2 = torch.nn.Conv2d(layers[1], layers[2], 5)
        self.fc1 = torch.nn.Linear(5 * 5 * layers[2], layers[3])
        self.fc2 = torch.nn.Linear(layers[3], layers[4])
        self.fc3 = torch.nn.Linear(layers[4], layers[5])
        self.layers = torch.nn.ModuleList([self.conv1, self.conv2, self.fc1, self.fc2, self.fc3])

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for i, l in enumerate(self.layers):
                if i == 0 and self.first_layer_normal:
                    l.weight.data.normal_()
                else:
                    self.sampler_w(l.weight.data)
                self.sampler_b(l.bias.data)

                l.weight.data.mul_(self.sigma_w)
                l.bias.data.mul_(self.sigma_b)

                if not self.scaling:
                    l.weight.data.div_(np.sqrt(self.layer_in_size(l))) 

    def layer_in_size(self, l):
        size = 1
        for sz in l.weight.size()[1:]:
            size *= sz
        return size

    def forward(self, x):
        if self.scaling:
            x = x / np.sqrt(self.layer_in_size(self.conv1))
        x = self.conv1(x)
        x = self.act_function(x)
        x = torch.nn.functional.max_pool2d(x, 2)

        if self.scaling:
            x = x / np.sqrt(self.layer_in_size(self.conv2))
        x = self.conv2(x)
        x = self.act_function(x)
        x = torch.nn.functional.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)

        if self.scaling:
            x = x / np.sqrt(self.layer_in_size(self.fc1))
        x = self.fc1(x)
        x = self.act_function(x)

        if self.scaling:
            x = x / np.sqrt(self.layer_in_size(self.fc2))
        x = self.fc2(x)
        x = self.act_function(x)

        if self.scaling:
            x = x / np.sqrt(self.layer_in_size(self.fc3))
        x = self.fc3(x)
        x = torch.nn.functional.log_softmax(x, dim = 1)

        return x
