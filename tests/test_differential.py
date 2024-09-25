import torch
import grnewt

def test_taylor_n_scalar_linear():
    dtype = torch.float64

    a = torch.nn.Parameter(torch.randn(1, dtype = dtype))
    b = torch.nn.Parameter(torch.randn(1, dtype = dtype))
    x = torch.randn(1, dtype = dtype)
    y = a * x + b
    v = torch.randn(1, dtype = dtype)
    result = grnewt.taylor_n(y, (a, b), 1, [v])

    assert torch.allclose(result, torch.tensor([y, x * v], dtype = dtype))

def test_taylor_n_scalar_nonlinear():
    dtype = torch.float64

    N = 7
    a = torch.nn.Parameter(torch.randn(1, dtype = dtype))
    lst_a = torch.concat([a.pow(i) for i in range(N + 1)])
    lst_x = torch.randn(N + 1, dtype = dtype)
    y = torch.dot(lst_a, lst_x)
    v = torch.randn(1, dtype = dtype)
    result = grnewt.taylor_n(y, (a,), N, [v] * N)

    expected = torch.empty(N + 1, dtype = dtype)
    factors = lst_x.clone()
    powers = torch.tensor([i for i in range(N + 1)], dtype = dtype)
    with torch.no_grad():
        for i in range(N + 1):
            expected[i] = sum([f * a.pow(p) for f, p in zip(factors, powers)])
            for i in range(N + 1):
                factors[i] *= powers[i]
                if powers[i] > 0:
                    powers[i] -= 1

    for i in range(N + 1):
        expected[i] = expected[i] * v.pow(i)

    assert torch.allclose(result, expected)
