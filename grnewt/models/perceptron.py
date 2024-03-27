import numpy as np
import torch
from .. import partition as build_partition

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

    def partition(self, partition_args):
        idx_conv2d = list(range(len(self.layers)))
        nlayers = len(idx_conv2d)
        if partition_args.find('blocks') == 0:
            def fn_blocks(size, nblocks):
                q = size // nblocks
                r = size % nblocks
                return [q + 1 for i in range(r)] + [q for i in range(nblocks - r)]

            n = int(partition_args[len('blocks-'):])
            lst_blocks = fn_blocks(nlayers, n)
            pre_groups = []
            k = 0
            for bsize in lst_blocks:
                gr = []
                for i in range(bsize):
                    gr.append(idx_conv2d[k + i])
                pre_groups.append(gr)
                k += bsize

            lst_names_w = [['layers.{}.weight'.format(k) for k in gr] for gr in pre_groups]
            lst_names_b = [['layers.{}.bias'.format(k) for k in gr] for gr in pre_groups]

            param_groups, name_groups = build_partition.names_by_lst(self, lst_names_w + lst_names_b)
        elif partition_args.find('alternate') == 0:
            n = int(partition_args[len('alternate-'):])
            lst_names_w = [['layers.{}.weight'.format(k) for i, k in enumerate(idx_conv2d) if i % n == r] for r in range(n)]
            lst_names_b = [['layers.{}.bias'.format(k) for i, k in enumerate(idx_conv2d) if i % n == r] for r in range(n)]

            param_groups, name_groups = build_partition.names_by_lst(self, lst_names_w + lst_names_b)
        else:
            NotImplementedError('Error: not implemented partition_args: "{}".'.format(partition_args))

        return param_groups, name_groups
