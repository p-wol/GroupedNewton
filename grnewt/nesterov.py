import warnings
import numpy as np
import scipy
import torch

def nesterov_lrs(H, g, order3, *, damping_int = 1.):
    """
    Computes learning rates with "anisotropic Nesterov" cubic regularization.
    Let: D = order3.abs().pow(1/3).diag()
    Returns a vector lrs such that:
        lrs = (H + .5 * damping_int * ||D lrs||_2 D^2)^{-1} g.
    Additional variables:
        lrs_prime = D @ lrs,
        r = ||lrs_prime||.
    Look first for a scalar r such that:
        r = ||(H @ D_inv + .5 * damping_int * r * D)^{-1} g||, (1)
    then compute lrs_prime, and finally lrs.
    """
    # Define some useful variables
    D_vec = order3.abs().pow(1/3)
    D = D_vec.diag()
    D_inv = (1/D_vec).diag()
    try:
        Hd = torch.linalg.eigh(H).eigenvalues
        H_pd = ((Hd <= 0).sum() == 0)    # boolean, True if H is Positive Definite
    except:
        warnings.warn("pytorch.linalg.eigh failed on the Hessian summary matrix H, \
                switching to default behavior.")
        print('Warning "torch.linalg.eigh(H)": H = {}'.format(H))
        H_pd = True

    # Function whose fixed points should be found
    def f(x):
        try:
            return torch.linalg.solve(H @ D_inv + .5 * damping_int * x * D, g).norm().item() - x
        except:
            return np.inf

    # Function Returning an upper bound on the solution r of (1)
    def compute_r_max():
        cond1 = (2/damping_int) * (D @ g).norm().item()
        cond2 = ((2/damping_int)**2 * (D @ H @ D.inverse().pow(2) @ g)).norm().sqrt().item()
        return max(cond1, cond2)

    # Function computing 'lrs' from 'r'
    def compute_lrs(x): 
        return D_inv @ torch.linalg.solve(H @ D_inv + .5 * damping_int * x * D, g)

    # Compute x0
    if H_pd:
        x0 = 0
    else:
        lambd_min = torch.linalg.eigh(D_inv @ H @ D_inv).eigenvalues.min().item()
        lambd_min = abs(min(0, lambd_min))
        x0 = (2/damping_int) * lambd_min

    # Compute x1
    x1 = compute_r_max()
    i = 0
    while f(x1) >= 0:
        x1 *= 3
        i += 1
        if i >= 20:
            raise RuntimeError('Impossible to find a solution to Nesterov problem.')

    # Compute lrs
    r = scipy.optimize.root_scalar(f, bracket = [x0, x1], maxiter = 100)
    return compute_lrs(r.root), r.root, r.converged

