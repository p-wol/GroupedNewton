import numpy as np
import torch
import torch.nn as nn
from .. import partition as build_partition

class VGG(nn.Module):
    def __init__(self, cfg_type, fc_sizes = [4096, 4096], image_size = 224, num_classes = 1000,
            scaling = False, sigma_w = 1., name_act_function = 'relu', batch_norm = False):
        super().__init__()

        self.scaling = scaling
        self.sigma_w = sigma_w
        fc_multiplier = image_size // 32

        dct_act_functions = {'identity': nn.Identity, 
                             'tanh': nn.Tanh, 
                             'relu': nn.ReLU, 
                             'sigmoid': nn.Sigmoid, 
                             'elu': nn.ELU}
        cl_act_function = dct_act_functions[name_act_function]

        self.features = make_layers(cfgs[cfg_type], cl_act_function, batch_norm = batch_norm)

        self.avgpool = nn.AdaptiveAvgPool2d((fc_multiplier, fc_multiplier))

        fc_sizes = fc_sizes + [num_classes]
        last_sz = fc_sizes[0]
        self.classifier = nn.ModuleList([nn.Linear(512 * fc_multiplier**2, last_sz)])
        for fc_sz in fc_sizes[1:]:
            self.classifier += [cl_act_function(inplace = True), nn.Linear(last_sz, fc_sz)]
            last_sz = fc_sz

        with torch.no_grad():
            for m in self.modules():
                if type(m) in (nn.Linear, nn.Conv2d):
                    m.weight.data.normal_()
                    m.bias.data.zero_()

                    m.weight.data.mul_(self.sigma_w)
                    if not self.scaling:
                        m.weight.data.div_(np.sqrt(self.layer_in_size(m)))

    def layer_in_size(self, l):
        size = 1
        for sz in l.weight.size()[1:]:
            size *= sz
        return size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for m in self.features:
            if type(m) in (nn.Linear, nn.Conv2d) and self.scaling:
                x = x / np.sqrt(self.layer_in_size(m))
            x = m(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        for m in self.classifier:
            if type(m) in (nn.Linear, nn.Conv2d) and self.scaling:
                x = x / np.sqrt(self.layer_in_size(m))
            x = m(x)

        x = nn.functional.log_softmax(x, dim = 1)

        return x

    def partition(self, partition_args):
        idx_conv2d = [i for i, k in enumerate(self.features) if isinstance(k, nn.Conv2d)]
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

            lst_names_w = [['features.{}.weight'.format(k) for k in gr] for gr in pre_groups]
            lst_names_b = [['features.{}.bias'.format(k) for k in gr] for gr in pre_groups]

            lst_names_w.append(['classifier.0.weight'])
            lst_names_b.append(['classifier.0.bias'])
            param_groups, name_groups = build_partition.names_by_lst(self, lst_names_w + lst_names_b)
        elif partition_args.find('alternate') == 0:
            n = int(partition_args[len('alternate-'):])
            lst_names_w = [['features.{}.weight'.format(k) for i, k in enumerate(idx_conv2d) if i % n == r] for r in range(n)]
            lst_names_b = [['features.{}.bias'.format(k) for i, k in enumerate(idx_conv2d) if i % n == r] for r in range(n)]

            lst_names_w.append(['classifier.0.weight'])
            lst_names_b.append(['classifier.0.bias'])
            param_groups, name_groups = build_partition.names_by_lst(self, lst_names_w + lst_names_b)
        else:
            NotImplementedError('Error: not implemented partition_args: "{}".'.format(partition_args))

        return param_groups, name_groups


def make_layers(cfg, cl_act_function, batch_norm = False):
    layers = nn.ModuleList()
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size = 2, stride = 2))
        else:
            v = int(v)
            layers.append(nn.Conv2d(in_channels, v, kernel_size = 3, padding = 1))
            if batch_norm: 
                layers.append(nn.BatchNorm2d(v))
            layers.append(cl_act_function(inplace = True))
            in_channels = v
    return layers


cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}
