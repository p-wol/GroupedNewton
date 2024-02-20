import warnings
import numpy as np
import scipy
import torch

#TODO: * put threshold_D_sing in args
#      * add warning when not converged
def nesterov_lrs(H, g, order3_, *, damping_int = 1., force_numerical_x0 = False, threshold_D_sing = 1e-4):
    """
    Computes learning rates with "anisotropic Nesterov" cubic regularization.
    Let: D = order3_.diag()
    Returns a vector lrs such that:
        lrs = (H + .5 * damping_int * ||D lrs||_2 D^2)^{-1} g.
    Additional variables:
        lrs_prime = D @ lrs,
        r = ||lrs_prime||.
    Look first for a scalar r such that:
        r = ||(H @ D_inv + .5 * damping_int * r * D)^{-1} g||, (1)
    then compute lrs_prime, and finally lrs.

    Arguments:
     * H: summary of the Hessian (in R^(S * S))
     * g: summary of the gradient (in R^S)
     * order3_: vector such that D = order3_.diag()
     * damping_int: internal damping, also called lambda_int
     * force_numerical_x0: when H is not positive definite, r must be searched 
       in [x0, infinity], where x0 is to be computed:
        1) if D is not singular, x0 can be computed with a formula,
        2) if D is singular (or has a close-to-zero diagonal value), a numerical
          computation via scipy.optimie.root_scalar becomes necessary;
       if force_numerical_x0 is True, option 2 is always chosen.
     * threshold_D_sing: if a diagonal value of D is below this threshold, then
       D is considered as singular and option 2 is chosen to compute x0.
    """
    device = H.device
    dct_logs = {}

    # Define some useful variables
    D_vec = order3_
    D = D_vec.diag()
    D_squ = D.pow(2)

    # Check if H is positive definite
    Hd = torch.linalg.eigh(H).eigenvalues
    #print('eigs: ', Hd)
    H_pd = ((Hd <= 0).sum() == 0).item()
    dct_logs['H_pd'] = H_pd

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
    if force_numerical_x0:
        D_sing = True
    else:
        D_sing = ((D_vec <= threshold_D_sing).sum() > 0).item()
    dct_logs['D_sing'] = D_sing
    if not D_sing:
        D_inv = D.inverse()

    # Function whose roots should be found
    def f(x):
        try:
            return (D @ torch.linalg.solve(H + .5 * damping_int * x * D_squ, g)).norm().item() - x
        except:
            return np.inf

    # Function whose root should be found to compute the minimal r
    #TODO: explain
    def fn_g(x, H64, D_squ64):
        x = torch.tensor(x).squeeze()
        if x.dim() == 0:
            return torch.linalg.eigh(H64 + .5 * damping_int * x * D_squ64)\
                    .eigenvalues.min().item() 
        else:
            NotImplementedError('Optimization of fn_g involves a non-scalar np.ndarray. Exiting.')

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
        dct_logs['x0'] = x0
    else:
        # Case H non positive definite
        # Computation of the largest value r = x0 for which the matrix to invert (H + .5 * damping_int * r * D_squ) is singular
        if not D_sing:
            # D non singular: explicit computation
            lambd_min = torch.linalg.eigh(D_inv @ H @ D_inv).eigenvalues.min().item()
            lambd_min = abs(min(0, lambd_min))
            x0 = (2/damping_int) * lambd_min
            dct_logs['x0'] = x0
        else:
            # D singular: use root finding
            H64 = H #.to(device = torch.device('cpu'), dtype = torch.float64)
            D_squ64 = D_squ #.to(device = torch.device('cpu'), dtype = torch.float64)
            gx0 = 0. # torch.tensor(0., dtype = torch.float64).numpy() # -torch.linalg.eigh(H64).eigenvalues.min().numpy()
            gx1 = 1. #torch.tensor(1., dtype = torch.float64).numpy() #gx0 + torch.tensor(1e9, dtype = torch.float64).numpy()
            print('g(x0) =', fn_g(gx0, H64, D_squ64))
            while fn_g(gx1, H64, D_squ64) <= 0:
                print('looking for gx1: x1 = {}, g(x1) = {}'.format(gx1, fn_g(gx1, H64, D_squ64)))
                gx1 *= 3
            #print('gx0 = {}, gx1 = {}'.format(gx0, gx1))
            #print('g(gx0) = {}, g(gx1) = {}'.format(fn_g(gx0, H64, D_squ64), fn_g(gx1, H64, D_squ64)))
            r = scipy.optimize.root_scalar(lambda x: fn_g(x, H64, D_squ64), bracket = [gx0, gx1], maxiter = 100)
            #print(r)
            dct_logs['x0'] = r.root
            dct_logs['x0_converged'] = r.converged
            if not r.converged:
                return None, dct_logs
                #raise RuntimeError('H has at least one negative eigenvalue, D is singular, and no solution was found to regularize H.')
            x0 = r.root
            #print('before: g(x0) = {}, f(x0) = {}'.format(fn_g(x0, H64, D_squ64), f(x0)))
            while f(x0) <= 0:
                #print('reducing: g(x0) = {}, f(x0) = {}'.format(fn_g(x0), f(x0)))
                x0 *= .99


    # Compute x1
    x1 = x0 + 1. #compute_r_max()
    i = 0
    while f(x1) >= 0:
        x1 *= 3
        i += 1
        if i >= 40:
            raise RuntimeError('Impossible to find a solution to Nesterov problem.')
    dct_logs['x1'] = x1

    # Compute lrs
    r = scipy.optimize.root_scalar(f, bracket = [x0, x1], maxiter = 100, rtol = 1e-4)
    dct_logs['r'] = r.root
    dct_logs['r_converged'] = r.converged
    return compute_lrs(r.root), dct_logs
