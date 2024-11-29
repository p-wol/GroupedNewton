import copy
import pytest
import torch
import grnewt
from grnewt import partition as build_partition
from grnewt import compute_Hg, compute_Hg_fullbatch, ParamGroups

class Polynomial(torch.nn.Module):
    def __init__(self, degree):
        super(Polynomial, self).__init__()
        
        self.coeffs = torch.nn.ParameterList()
        for d in range(degree + 1):
            self.coeffs.append(torch.nn.Parameter(torch.tensor(0.).normal_()))
        
    def forward(self, x):
        s = 0
        for d, coeff in enumerate(self.coeffs):
            s = s + x[:,d] * coeff.pow(d)
        return s

@pytest.fixture
def degree():
    return 7

@pytest.fixture
def polynomial(degree):
    return Polynomial(degree)

def _compute_Hg_pytorch(full_loss, x, y, tup_params):
    D = len(tup_params)

    # Order 1
    loss = full_loss(x, y)
    g2_tup = torch.autograd.grad(loss, tup_params, create_graph = True, materialize_grads = True)
    g2 = torch.stack([g.detach() for g in g2_tup])

    # Order 2
    H2 = torch.zeros(D, D)
    deriv_i = [None] * D
    for i, g in enumerate(g2_tup):
        if not g.requires_grad:
            h = tuple(torch.tensor(0.) for j in range(D))
        else:
            h = torch.autograd.grad(g, tup_params, create_graph = True, materialize_grads = True)

        for j in range(D):
            H2[i,j] = h[j].detach()
            H2[j,i] = h[j].detach()
        deriv_i[i] = h[i]

    # Order 3
    order32 = [None] * D
    for i, der in enumerate(deriv_i):
        if not der.requires_grad:
            der_ = tuple(torch.tensor(0.) for j in range(D))
        else:
            der_ = torch.autograd.grad(der, tup_params, create_graph = True, materialize_grads = True)

        order32[i] = der_[i].detach()

    order32 = torch.stack(order32)

    return H2, g2, order32

def test_Hg_polynomial_canonical(polynomial, degree):
    batch_size = 10
    x = torch.randn(batch_size, degree + 1)
    y = torch.randn(batch_size, 1)

    pgroups, name_groups = build_partition.canonical(polynomial)
    param_groups = ParamGroups(pgroups)

    def full_loss(x_, y_): 
        return polynomial(x_).sum()

    # Compute the derivatives with the custom functions of grnewt
    direction = tuple([torch.tensor(1.)] * (degree + 1))
    H1, g1, order31 = compute_Hg(param_groups, full_loss, x, y, direction)

    # Compute the derivatives with pytorch
    H2, g2, order32 = _compute_Hg_pytorch(full_loss, x, y, param_groups.tup_params)

    assert torch.allclose(g1, g2)
    assert torch.allclose(H1, H2)
    assert torch.allclose(order31, order32)

def test_Hg_polynomial_canonical_with_loss(polynomial, degree):
    batch_size = 10
    x = torch.randn(batch_size, degree + 1)
    y = torch.randn(batch_size, 1)

    pgroups, name_groups = build_partition.canonical(polynomial)
    param_groups = ParamGroups(pgroups)

    def full_loss(x_, y_): 
        return (polynomial(x_) - y_).pow(2).mean()

    # Compute the derivatives with the custom functions of grnewt
    direction = tuple([torch.tensor(1.)] * (degree + 1))
    H1, g1, order31 = compute_Hg(param_groups, full_loss, x, y, direction)

    # Compute the derivatives with pytorch
    H2, g2, order32 = _compute_Hg_pytorch(full_loss, x, y, param_groups.tup_params)

    assert torch.allclose(g1, g2)
    assert torch.allclose(H1, H2)
    assert torch.allclose(order31, order32)

def test_Hg_polynomial_trivial(polynomial, degree):
    batch_size = 10
    x = torch.randn(batch_size, degree + 1)
    y = torch.randn(batch_size, 1)

    pgroups, name_groups = build_partition.trivial(polynomial)
    param_groups = ParamGroups(pgroups)

    def full_loss(x_, y_): 
        return polynomial(x_).sum()

    # Compute the derivatives with the custom functions of grnewt
    direction = tuple([torch.tensor(1.)] * (degree + 1))
    H1, g1, order31 = compute_Hg(param_groups, full_loss, x, y, direction)

    # Compute the derivatives with pytorch
    H2, g2, order32 = _compute_Hg_pytorch(full_loss, x, y, param_groups.tup_params)
    print(order31, order32)
    g2 = torch.sum(g2, (0,), keepdim = True)
    H2 = torch.sum(H2, (0, 1), keepdim = True)
    order32 = torch.sum(order32, (0,), keepdim = True)


    assert torch.allclose(g1, g2)
    assert torch.allclose(H1, H2)
    assert torch.allclose(order31, order32)

def test_Hg_polynomial_trivial_with_loss(polynomial, degree):
    batch_size = 10
    x = torch.randn(batch_size, degree + 1)
    y = torch.randn(batch_size, 1)

    pgroups, name_groups = build_partition.trivial(polynomial)
    param_groups = ParamGroups(pgroups)

    def full_loss(x_, y_): 
        return (polynomial(x_) - y_).pow(2).sum()

    # Compute the derivatives with the custom functions of grnewt
    direction = tuple([torch.tensor(1.)] * (degree + 1))
    print("==========")
    H1, g1, order31 = compute_Hg(param_groups, full_loss, x, y, direction)
    print("==========")
    
    H1p, g1p, order31p = compute_Hg(ParamGroups(build_partition.canonical(polynomial)[0]), full_loss, x, y, direction)
    print("raw Hg: ", order31p)

    # Compute the derivatives with pytorch
    H2, g2, order32 = _compute_Hg_pytorch(full_loss, x, y, param_groups.tup_params)
    print(order31, order32)
    g2 = torch.sum(g2, (0,), keepdim = True)
    H2 = torch.sum(H2, (0, 1), keepdim = True)
    order32 = torch.sum(order32, (0,), keepdim = True)

    assert torch.allclose(g1, g2)
    assert torch.allclose(H1, H2)
    assert torch.allclose(order31, order32)

