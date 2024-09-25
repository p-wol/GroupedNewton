import copy
import pytest
import torch
from torch import nn
import grnewt
from grnewt.optimizers import AdamUpdate, SGDUpdate
from torch.optim import Adam, SGD

class Perceptron(torch.nn.Module):
    def __init__(self, layers, act_name = 'tanh'):
        super(Perceptron, self).__init__()
        
        if act_name == 'identity':
            act_name = 'linear'
    
        gain = nn.init.calculate_gain(act_name)
        
        self.layers = torch.nn.ModuleList()
        for l_in, l_out in zip(layers[:-1], layers[1:]):
            self.layers.append(torch.nn.Linear(l_in, l_out))
            with torch.no_grad():
                self.layers[-1].weight.mul_(gain)
        self.nb_layers = len(self.layers)
        
        if act_name in ['tanh', 'sigmoid', 'relu']:
            self.act_function = torch.__dict__[act_name]
        elif act_name == 'linear':
            self.act_function = lambda x: x
        
    def forward(self, x):
        for l in self.layers[:-1]:
            x = l(x)
            x = self.act_function(x)
        x = self.layers[-1](x)
        return x

def check_equal(m1, m2):
    e = True
    
    dct1 = dict(m1.named_parameters())
    dct2 = dict(m2.named_parameters())
    for n, p in dct1.items():
        e &= torch.allclose(p, dct2[n])
    return e

@pytest.fixture
def dataset(model):
    num_batches = 5

    n_tr = 7
    x_tr = [torch.randn(n_tr, model.layers[0].in_features) for i in range(num_batches)]
    y_tr = [torch.randn(n_tr, model.layers[-1].out_features) for i in range(num_batches)]

    return x_tr, y_tr

@pytest.fixture
def model():
    layers = [100, 60, 20, 10]
    act_name = 'tanh'

    return Perceptron(layers, act_name)

def _test_optim(model, dataset, Cl_Update, Cl_Optim, **kwargs):
    torch.set_default_dtype(torch.float64)

    # Define loss, dataset and models
    loss_mean = torch.nn.MSELoss(reduction = 'mean')
    x_tr, y_tr = dataset
    model1 = model
    model2 = copy.deepcopy(model)

    updater = Cl_Update(model1.parameters(), **kwargs)
    optimizer = Cl_Optim(model2.parameters(), **kwargs)

    epochs = 10
    num_batches = len(x_tr)

    e = True
    for i in range(epochs * num_batches):
        x = x_tr[i % num_batches]
        y = y_tr[i % num_batches]

        # First model
        model1.zero_grad()
        f1 = loss_mean(model1(x), y)
        f1.backward()

        update = updater.compute_step()
        updater.step(update)

        # Second model
        model2.zero_grad()
        f2 = loss_mean(model2(x), y)
        f2.backward()

        optimizer.step()

        # Final test
        e &= check_equal(model1, model2)

    return e

def test_adam(model, dataset):
    assert _test_optim(model, dataset, AdamUpdate, Adam, lr = 1e-3)

def test_sgd(model, dataset):
    assert _test_optim(model, dataset, SGDUpdate, SGD, lr = 1e-3)

