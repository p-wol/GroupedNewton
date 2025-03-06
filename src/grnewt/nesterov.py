import warnings
import time
import numpy as np
import scipy
import torch

#TODO: * put threshold_D_sing in args
#      * add warning when not converged
def nesterov_lrs(H, g, order3_, *, damping_int = 1., force_x0_computation = None, threshold_D_sing = 1e-5):
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
    time_beginning = time.time()
    
    device = H.device
    dtype = H.dtype
    dct_logs = {}

    # Create useful variables
    H = H.to(dtype = torch.float64)
    g = g.to(dtype = torch.float64)
    order3_ = order3_.to(dtype = torch.float64)
    D = order3_.diag()
    D_squ = order3_.pow(2).diag()
    D_inv = (1/order3_).diag()

    # Function whose roots should be found, between x0 and x1
    def f(x):
        return (D @ torch.linalg.solve(H + .5 * damping_int * x * D_squ, g)).norm().item() - x

    # Compute x0
    x0, dct_logs_x0 = compute_x0(H, order3_, D_squ, D_inv, damping_int, \
        threshold_D_sing = threshold_D_sing, force_x0_computation = force_x0_computation)
    for k, v in dct_logs_x0.items():
        dct_logs['x0.' + k] = v

    # Check the validity of x0
    if x0 is None:
        dct_logs['found'] = False
        dct_logs['time'] = time.time() - time_beginning
        return None, dct_logs
    try:
        val_fx0 = f(x0)
    except:
        x0 *= 1.001
        val_fx0 = f(x0)
    dct_logs['x0'] = x0
    dct_logs['f(x0) > 0'] = (val_fx0 > 0)
    if val_fx0 <= 0:
        dct_logs['found'] = False
        dct_logs['time'] = time.time() - time_beginning
        return None, dct_logs

    # Compute x1
    x1 = x0 + 1.
    i = 0
    while f(x1) >= 0:
        x1 *= 3
        i += 1
    dct_logs['x1'] = x1

    # Compute lrs
    r = scipy.optimize.root_scalar(f, bracket = [x0, x1], maxiter = 100) #, rtol = 1e-4)
    r_root = torch.tensor(r.root, dtype = torch.float64, device = device)
    dct_logs['r'] = r_root
    dct_logs['r_converged'] = r.converged
    dct_logs['found'] = r.converged

    # Compute 'lrs' from 'r'
    lrs = torch.linalg.solve(H + .5 * damping_int * r_root * D_squ, g)
    dct_logs['lrs'] = lrs
    
    dct_logs['time'] = time.time() - time_beginning
    return lrs.to(dtype = dtype), dct_logs

def compute_x0(H, order3_, D_squ, D_inv, damping_int, \
        threshold_D_sing = 1e-5, force_x0_computation = None):
    dct_logs = {}

    # Check if H is positive definite
    Hd = torch.linalg.eigvalsh(H)

    H_pd = ((Hd <= 0).sum() == 0).item()
    dct_logs['H_pd'] = H_pd

    # Error if H is not positive definite and damping_int == 0 (case impossible to solve)
    if not H_pd and damping_int == 0:
        raise ValueError('H is not positive definite and damping_int == 0: case impossible to solve. You may try damping_int > 0.')

    # Determine how to compute x0
    dct_logs['D_sing'] = 'None'
    if force_x0_computation is None:
        if H_pd:                # H positive definite (PD)
            x0_computation = 'Direct_Hpd'
        else:
            # Check if D is singular
            D_sing = ((order3_ <= threshold_D_sing).sum() > 0).item()
            dct_logs['D_sing'] = '{}'.format(D_sing)

            if not D_sing:      # H not PD and D not singular
                x0_computation = 'Analytical'
            else:               # H not PD and D singular
                x0_computation = 'Numerical'
    else:
        if force_x0_computation in ['Direct_Hpd', 'Analytical', 'Numerical']:
            x0_computation = force_x0_computation
        else:
            raise ValueError('Error: unknown value for "force_x0_computation", found {}.'.format(force_x0_computation))

    # Compute x0
    dct_logs['computation'] = x0_computation
    if x0_computation == 'Direct_Hpd':
        # Case H positive definite (PD)
        x0 = 0.
        dct_logs['found'] = True
    elif x0_computation == 'Analytical':
        # Case H not PD and D not singular
        # Computation of the largest value r = x0 for which the matrix to invert (H + .5 * damping_int * r * D_squ) is singular

        lambd_min = torch.linalg.eigvalsh(D_inv @ H @ D_inv).min().item()
        lambd_min = abs(min(0, lambd_min))
        x0 = (2/damping_int) * lambd_min
        dct_logs['found'] = True
    elif x0_computation == 'Numerical':
        # Case H not PD and D singular
        # D singular: use root finding

        # Function whose root should be found to compute x0
        #TODO: explain
        def fn_g(x):
            return torch.linalg.eigvalsh(H + .5 * damping_int * x * D_squ).min().item()
    
        gx0 = 0.
        gx1 = 1.
        last_g = fn_g(gx1)
        while last_g <= 0:
            gx1 *= 3
            curr_g = fn_g(gx1)
            if curr_g < last_g:
                x0 = None
                dct_logs['found'] = False
                dct_logs['computation'] = 'Numer_dvx1'
                return x0, dct_logs
            else:
                last_g = curr_g

        rx0 = scipy.optimize.root_scalar(fn_g, bracket = [gx0, gx1], maxiter = 200)
        
        if rx0.converged:
            x0 = rx0.root
            dct_logs['found'] = True
            dct_logs['computation'] = 'Numer_conv'
        else:
            x0 = None
            dct_logs['found'] = False
            dct_logs['computation'] = 'Numer_divg'
    else:
        raise NotImplementedError('Unsupported case x0_computation == {}.'.format(x0_computation))

    return x0, dct_logs
