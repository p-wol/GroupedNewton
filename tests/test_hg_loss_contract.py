"""The contract of the `loss` argument of `compute_Hg` / `compute_Hg_batched`.

These functions used to take `(full_loss, x, y)` and build the graph themselves.
Taking an already-evaluated tensor is the better interface, but it moves a
contract that used to be structurally guaranteed onto the caller, where it is
invisible in the signature. This file pins that contract.
"""

import pytest
import torch

from grnewt import (
    ParamStructure,
    compute_Hg,
    compute_Hg_batched,
)
from grnewt import (
    partition as build_partition,
)


def _setup(n=24, seed=0):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(5, 6), torch.nn.Tanh(), torch.nn.Linear(6, 3))
    x = torch.randn(n, 5)
    y = torch.randint(0, 3, (n,))
    loss_fn = torch.nn.CrossEntropyLoss()

    def full_loss(a, b):
        return loss_fn(model(a), b)

    ps = ParamStructure(build_partition.canonical(model)[0])
    u = tuple(torch.randn_like(p) for p in ps.tup_params)
    return model, x, y, full_loss, ps, u


def _pre(x, y):
    return x, y


IMPLEMENTATIONS = {"reference": compute_Hg, "batched": compute_Hg_batched}


# ---------------------------------------------------------------------------
# the contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(IMPLEMENTATIONS))
def test_a_loss_without_a_graph_raises_instead_of_returning_zeros(f64, name):
    """The dangerous case. `ParamStructure.dercon` returns zeros when its input
    does not require grad -- right for an inner call on a group the loss does not
    depend on, catastrophic for the top-level loss. Before the guard, a stray
    `.detach()` made compute_Hg return Hbar = gbar = order3 = 0, nesterov_lrs
    return lrs = 0, and the run proceed without ever moving: everything finite,
    nothing raised, every log plausible."""
    fn = IMPLEMENTATIONS[name]
    _, x, y, full_loss, ps, u = _setup()

    with pytest.raises(RuntimeError, match="does not require grad"):
        fn(ps, full_loss(x, y).detach(), u)

    with torch.no_grad():
        loss_ng = full_loss(x, y)
    with pytest.raises(RuntimeError, match="does not require grad"):
        fn(ps, loss_ng, u)


@pytest.mark.parametrize("name", sorted(IMPLEMENTATIONS))
def test_a_callable_is_refused_by_name(f64, name):
    fn = IMPLEMENTATIONS[name]
    _, _, _, full_loss, ps, u = _setup()
    with pytest.raises(TypeError, match="not a callable"):
        fn(ps, full_loss, u)


@pytest.mark.parametrize("name", sorted(IMPLEMENTATIONS))
def test_a_non_scalar_loss_is_refused(f64, name):
    """A per-sample loss would otherwise die inside autograd with
    'grad can be implicitly created only for scalar outputs'."""
    fn = IMPLEMENTATIONS[name]
    model, x, _, _, ps, u = _setup()
    per_sample = ((model(x) - torch.randn(24, 3)) ** 2).mean(dim=1)
    with pytest.raises(ValueError, match="must be a scalar"):
        fn(ps, per_sample, u)


@pytest.mark.parametrize("name", sorted(IMPLEMENTATIONS))
def test_a_valid_loss_is_untouched_by_the_guard(f64, name):
    """The guard must not change any accepted value."""
    fn = IMPLEMENTATIONS[name]
    _, x, y, full_loss, ps, u = _setup()
    h, g, o3 = fn(ps, full_loss(x, y), u)
    h_ref, g_ref, o3_ref = compute_Hg(ps, full_loss(x, y), u)
    assert torch.allclose(h, h_ref, rtol=1e-11, atol=1e-13)
    assert torch.allclose(g, g_ref, rtol=1e-11, atol=1e-13)
    assert torch.allclose(o3, o3_ref, rtol=1e-10, atol=1e-12)
    assert float(h.norm()) > 0


def test_one_graph_can_feed_the_gradient_and_the_summaries(f64):
    """The reason the signature change is worth making: the caller evaluates the
    objective once and uses the same graph for `.grad` and for the summaries,
    instead of compute_Hg re-running the forward pass internally. `create_graph`
    keeps it alive for the second traversal."""
    _, x, y, full_loss, ps, u = _setup()
    loss = full_loss(x, y)

    grads = torch.autograd.grad(loss, ps.tup_params, create_graph=True)
    h, g, _ = compute_Hg(ps, loss, u)  # same graph, traversed again

    g_ref = ps.dot(tuple(t.detach() for t in grads), u)
    assert torch.allclose(g, g_ref, rtol=1e-11, atol=1e-13)
    assert float(h.norm()) > 0


def test_a_backward_that_frees_the_graph_is_reported(f64):
    """`loss.backward()` without retain_graph frees the buffers; the failure then
    comes from deep inside autograd. Documented here so the message is on record."""
    _, x, y, full_loss, ps, u = _setup()
    loss = full_loss(x, y)
    loss.backward()
    with pytest.raises(RuntimeError, match="backward through the graph a second time"):
        compute_Hg(ps, loss, u)
