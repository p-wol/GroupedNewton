import copy

import pytest
import torch
from torch.optim import SGD, Adam

from grnewt.optimizers import AdamUpdate, SGDUpdate


class Perceptron(torch.nn.Module):
    def __init__(self, layers, act_name="tanh"):
        super().__init__()

        if act_name == "identity":
            act_name = "linear"

        gain = torch.nn.init.calculate_gain(act_name)

        self.layers = torch.nn.ModuleList()
        for l_in, l_out in zip(layers[:-1], layers[1:]):
            self.layers.append(torch.nn.Linear(l_in, l_out))
            with torch.no_grad():
                self.layers[-1].weight.mul_(gain)
        self.nb_layers = len(self.layers)

        if act_name in ["tanh", "sigmoid", "relu"]:
            self.act_function = torch.__dict__[act_name]
        elif act_name == "linear":
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
        print(p - dct2[n])
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
    act_name = "tanh"

    return Perceptron(layers, act_name)


def _test_optim(model, dataset, Cl_Update, Cl_Optim, **kwargs):
    # Define loss, dataset and models
    loss_mean = torch.nn.MSELoss(reduction="mean")
    x_tr, y_tr = dataset
    model1 = model
    model2 = copy.deepcopy(model)

    updater = Cl_Update(model1.parameters(), **kwargs)
    optimizer = Cl_Optim(model2.parameters(), **kwargs)

    epochs = 10
    num_batches = len(x_tr)

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
        for n, p1 in model1.named_parameters():
            p2 = dict(model2.named_parameters())[n]
            assert torch.allclose(p1, p2), (
                f"diverged at step {i} on {n}: max|diff| = {(p1 - p2).abs().max().item():.3e}"
            )
    return True


def test_adam(f64, model, dataset):
    assert _test_optim(model, dataset, AdamUpdate, Adam, lr=1e-3)


def test_sgd(f64, model, dataset):
    assert _test_optim(model, dataset, SGDUpdate, SGD, lr=1e-3)


@pytest.mark.parametrize("Cl_Update", [SGDUpdate, AdamUpdate])
def test_step_is_a_descent_step(f64, model, dataset, Cl_Update):
    """Regression guard: updater.step() must DECREASE the loss.

    This is the invariant the sign bug violated. It does not depend on
    matching torch.optim, so it stays meaningful if the reference changes.
    """
    x, y = dataset[0][0], dataset[1][0]
    loss_fn = torch.nn.MSELoss()
    updater = Cl_Update(model.parameters(), lr=1e-3)

    model.zero_grad()
    before = loss_fn(model(x), y)
    before.backward()
    updater.step(updater.compute_step())

    with torch.no_grad():
        after = loss_fn(model(x), y)
    assert after.item() < before.item(), (
        f"loss increased: {before.item():.6e} -> {after.item():.6e}"
    )


@pytest.mark.parametrize("Cl_Update", [SGDUpdate, AdamUpdate])
def test_compute_step_returns_a_positive_direction(f64, model, dataset, Cl_Update):
    """Pins the convention NewtonSummary relies on: with lr=1 and no momentum,
    compute_step() returns +grad, and the caller supplies the minus sign."""
    x, y = dataset[0][0], dataset[1][0]
    model.zero_grad()
    torch.nn.MSELoss()(model(x), y).backward()

    updater = Cl_Update(model.parameters(), lr=1.0)
    direction = updater.compute_step()
    for p, d in zip(model.parameters(), direction):
        assert (p.grad * d).sum() > 0, "direction must be aligned with +grad"
