import numpy as np
import torch

class Perceptron(torch.nn.Module):
    def __init__(self, layers, act_function, scaling = False, sigma_w = 1., sigma_b = 1., \
            sampler_w = lambda t: t.normal_(), sampler_b = lambda t: t.normal_(), first_layer_normal = False,
            classification = True):
        super(Perceptron, self).__init__()

        self.act_function = act_function
        self.scaling = scaling
        self.sigma_w = sigma_w
        self.sigma_b = sigma_b
        self.sampler_w = sampler_w
        self.sampler_b = sampler_b
        self.first_layer_normal = first_layer_normal
        self.classification = classification

        self.layers = torch.nn.ModuleList()
        for l_in, l_out in zip(layers[:-1], layers[1:]):
            self.layers.append(torch.nn.Linear(l_in, l_out))
        self.nb_layers = len(self.layers)

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
                    l.weight.data.div_(np.sqrt(l.in_features))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for l in self.layers[:-1]:
            if self.scaling:
                x = x / np.sqrt(l.in_features)
            x = l(x)
            x = self.act_function(x)
        if self.scaling:
            x = x / np.sqrt(l.in_features)
        x = self.layers[-1](x)

        if self.classification:
            x = torch.nn.functional.log_softmax(x, dim = 1)

        return x
