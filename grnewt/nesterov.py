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
    D_squ = D.pow(2)

    # Check if H is positive definite
    Hd = torch.linalg.eigh(H).eigenvalues
    H_pd = ((Hd <= 0).sum() == 0).item()

    # Error if H is not positive definite and damping_int == 0 (case impossible to solve)
    if not H_pd and damping_int == 0:
        raise ValueError('H is not positive definite and damping_int == 0: case impossible to solve. You may try damping_int > 0.')
    """
    try:
        Hd = torch.linalg.eigh(H).eigenvalues
        H_pd = ((Hd <= 0).sum() == 0)    # boolean, True if H is Positive Definite
    except:
        warnings.warn("pytorch.linalg.eigh failed on the Hessian summary matrix H, \
                switching to default behavior.")
        print('Warning "torch.linalg.eigh(H)": H = {}'.format(H))
        H_pd = True
    """

    # Check if D is singular
    D_sing = ((D_vec == 0.).sum() > 0).item()
    if not D_sing:
        D_inv = D.inverse()

    # Function whose fixed points should be found
    def f(x):
        try:
            return (D @ torch.linalg.solve(H + .5 * damping_int * x * D_squ, g)).norm().item() - x
        except:
            return np.inf

    # Function returning an upper bound on the solution r of (1)
    def compute_r_max():
        cond1 = (2/damping_int) * (D @ g).norm().item()
        cond2 = ((2/damping_int)**2 * (D @ H @ D.inverse().pow(2) @ g)).norm().sqrt().item()
        return max(cond1, cond2)

    # Function computing 'lrs' from 'r'
    def compute_lrs(x): 
        return torch.linalg.solve(H + .5 * damping_int * x * D_squ, g)

    # Compute x0
    if H_pd:
        x0 = 0
    else:
        # Case H non positive definite
        # Computation of the largest value r = x0 for which the matrix to invert (H + .5 * damping_int * r * D_squ) is singular
        if not D_sing:
            # D non singular: explicit computation
            lambd_min = torch.linalg.eigh(D_inv @ H @ D_inv).eigenvalues.min().item()
            lambd_min = abs(min(0, lambd_min))
            x0 = (2/damping_int) * lambd_min
        else:
            # D singular: use root finding
            fn_g = lambda x: torch.linalg.eigh(H + .5 * damping_int * torch.tensor(x) * D_squ).eigenvalues.min().item()
            gx0 = -torch.linalg.eigh(H).eigenvalues.min().item()
            r = scipy.optimize.root_scalar(fn_g, x0 = gx0, maxiter = 100, rtol = 1e-4)
            if not r.converged:
                raise RuntimeError('H has at least one negative eigenvalue, D is singular, and no solution was found to regularize H.')
            x0 = r.root

    print('H_pd = ', H_pd)
    print('D_sing = ', D_sing)
    print('x0 = ', x0)

    # Compute x1
    x1 = x0 + 1. #compute_r_max()
    i = 0
    while f(x1) >= 0:
        print(f(x1))
        x1 *= 3
        i += 1
        if i >= 40:
            raise RuntimeError('Impossible to find a solution to Nesterov problem.')

    # Compute lrs
    r = scipy.optimize.root_scalar(f, bracket = [x0, x1], maxiter = 100)
    return compute_lrs(r.root), r.root, r.converged

