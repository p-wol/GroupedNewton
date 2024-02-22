import warnings
import time
import numpy as np
import scipy
import torch

#TODO: * put threshold_D_sing in args
#      * add warning when not converged
def nesterov_lrs(H, g, order3_, *, damping_int = 1., force_numerical_x0 = False, threshold_D_sing = 1e-5):
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

    def create_useful_variables(do_float64 = False):
        if not do_float64:
            return H, g, \
                order3_, order3_.diag(), order3_.pow(2).diag(), (1/order3_).diag()
        else:
            o3_ = order3_.to(dtype = torch.float64)
            return H.to(dtype = torch.float64), \
                g.to(dtype = torch.float64), \
                o3_, o3_.diag(), o3_.pow(2).diag(), (1/o3_).diag()

    # Check if the computation should be done with float64
    if order3_.min() < 1e-5 or torch.linalg.eigh(H).eigenvalues[0].abs() < 1e-5:
        do_float64 = True
        dct_logs['small_eigs'] = True
    else:
        do_float64 = False
        dct_logs['small_eigs'] = False
    dct_logs['do_float64'] = do_float64

    # Create useful variables
    H, g, order3_, D, D_squ, D_inv = create_useful_variables(do_float64)

    # Check if H is positive definite
    Hd = torch.linalg.eigh(H).eigenvalues
    H_pd = ((Hd <= 0).sum() == 0).item()
    dct_logs['H_pd'] = H_pd

    # Error if H is not positive definite and damping_int == 0 (case impossible to solve)
    if not H_pd and damping_int == 0:
        raise ValueError('H is not positive definite and damping_int == 0: case impossible to solve. You may try damping_int > 0.')

    # Check if D is singular
    if force_numerical_x0:
        D_sing = True
    else:
        D_sing = ((order3_ <= threshold_D_sing).sum() > 0).item()
    dct_logs['D_sing'] = D_sing

    # Function whose roots should be found
    def f(x):
        return (D @ torch.linalg.solve(H + .5 * damping_int * x * D_squ, g)).norm().item() - x

    # Compute x0
    if H_pd:
        x0 = 0
    else:
        # Case H non positive definite
        # Computation of the largest value r = x0 for which the matrix to invert (H + .5 * damping_int * r * D_squ) is singular
        if not D_sing:
            # D non singular: explicit computation
            def compute_x0_analytical(H, D_inv, damping_int):
                lambd_min = torch.linalg.eigh(D_inv @ H @ D_inv).eigenvalues.min().item()
                lambd_min = abs(min(0, lambd_min))
                return (2/damping_int) * lambd_min

            # Compute x0
            x0 = compute_x0_analytical(H, D_inv, damping_int)

            # Redo computation with float64 if not precise enough
            if f(x0) > 0:
                dct_logs['x0_analytical'] = 'float32'
            else:
                do_float64 = True
                dct_logs['do_float64'] = do_float64
                H, g, order3_, D, D_squ, D_inv = create_useful_variables(do_float64)

                x0 = compute_x0_analytical(H, D_inv, damping_int)

                dct_logs['x0_analytical'] = 'float64'
        else:
            # D singular: use root finding

            # Function whose root should be found to compute x0
            #TODO: explain
            def fn_g(x):
                return torch.linalg.eigh(H + .5 * damping_int * x * D_squ)\
                            .eigenvalues.min().item()
        
            gx0 = 0.
            gx1 = 1.
            while fn_g(gx1) <= 0:
                gx1 *= 3
            rx0 = scipy.optimize.root_scalar(fn_g, bracket = [gx0, gx1], maxiter = 200)
            
            if rx0.converged:
                dct_logs['x0_numerical'] = 'converged'
                x0 = rx0.root
            else:
                dct_logs['x0_numerical'] = 'not converged'
                dct_logs['found'] = False
                dct_logs['time'] = time.time() - time_beginning
                return None, dct_logs
    
    # Check the validity of x0
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
        """
        if i >= 40:
            raise RuntimeError('Impossible to find a solution to Nesterov problem.')
        """
    dct_logs['x1'] = x1

    # Compute lrs
    r = scipy.optimize.root_scalar(f, bracket = [x0, x1], maxiter = 100) #, rtol = 1e-4)
    dct_logs['r'] = r.root
    dct_logs['r_converged'] = r.converged
    dct_logs['found'] = r.converged

    # Function computing 'lrs' from 'r'
    def compute_lrs(x): 
        return torch.linalg.solve(H + .5 * damping_int * x * D_squ, g)
    lrs = compute_lrs(r.root).to(dtype = dtype)
    
    dct_logs['time'] = time.time() - time_beginning
    return lrs, dct_logs


#TODO: * put threshold_D_sing in args
#      * add warning when not converged
def nesterov_lrs_old(H, g, order3_, *, damping_int = 1., force_numerical_x0 = False, threshold_D_sing = 1e-5):
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
    dtype = H.dtype
    dct_logs = {}

    def convert_float64():
        # Returns H, g, D, D_squ, D_inv
        temp_order3_ = order3_.to(dtype = torch.float64)
        return H.to(dtype = torch.float64), \
               g.to(dtype = torch.float64), \
               temp_order3_.diag(), \
               temp_order3_.pow(2).diag(), \
               (1/temp_order3_).diag()

    # Check if the computation should be done with float64
    do_float64 = False
    if order3_.min() < 1e-5 or torch.linalg.eigh(H).eigenvalues[0].abs() < 1e-5:
        do_float64 = True
    dct_logs['do_float64'] = do_float64
    """
    else:
        D_inv_ = (1/order3_).diag()
        eigmin = torch.linalg.eigh(D_inv_ @ H @ D_inv_).eigenvalues[0]
        if eigmin < 0 and 1 / eigmin.abs() < 1e-5:
            do_float64 = True
    """

    #do_float64 = True

    if do_float64:
        H, g, D, D_squ, D_inv = convert_float64()
        #print('doing float64')
    else:
        D = order3_.diag()
        D_squ = D.pow(2)
        D_inv = (1/order3_).diag()

    # Check if H is positive definite
    Hd = torch.linalg.eigh(H).eigenvalues
    #print('Hd =', Hd)
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
        D_sing = ((order3_ <= threshold_D_sing).sum() > 0).item()
    dct_logs['D_sing'] = D_sing

    # Function whose roots should be found
    def f(x):
        """
        print(D.dtype)
        print(H.dtype)
        print(D_squ.dtype)
        print(g.dtype)
        print(type(x))
        """
        return (D @ torch.linalg.solve(H + .5 * damping_int * x * D_squ, g)).norm().item() - x
        """
        try:
            return (D @ torch.linalg.solve(H + .5 * damping_int * x * D_squ, g)).norm().item() - x
        except:
            return np.inf
        """

    # Function whose root should be found to compute the minimal r
    #TODO: explain
    def fn_g(x):
        return torch.linalg.eigh(H + .5 * damping_int * x * D_squ)\
                    .eigenvalues.min().item() 
        """
        x = torch.tensor(x).squeeze()
        if x.dim() == 0:
            return torch.linalg.eigh(H + .5 * damping_int * x * D_squ)\
                    .eigenvalues.min().item() 
        else:
            NotImplementedError('Optimization of fn_g involves a non-scalar np.ndarray. Exiting.')
        """
    
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

            # Redo computation with float64 if not precise enough
            if f(x0) <= 0:
                #print(' === WARNING === redo computation with float64')
                #print(' === f(x0) <= 0 ; f({}) = {} ==='.format(x0, f(x0)))
                do_float64 = True
                dct_logs['do_float64'] = do_float64
                H, g, D, D_squ, D_inv = convert_float64()

                lambd_min = torch.linalg.eigh(D_inv @ H @ D_inv).eigenvalues.min().item()
                lambd_min = abs(min(0, lambd_min))
                x0 = (2/damping_int) * lambd_min
        else:
            # D singular: use root finding
            gx0 = 0.
            gx1 = 1.
            #print('g(x0) =', fn_g(gx0))
            while fn_g(gx1) <= 0:
                #print(fn_g(gx1))
                #print('looking for gx1: x1 = {}, g(x1) = {}'.format(gx1, fn_g(gx1)))
                gx1 *= 3
            #print('gx0 = {}, gx1 = {}; g(gx0) = {}, g(gx1) = {}'.format(gx0, gx1, fn_g(gx0), fn_g(gx1)))
            
            r = scipy.optimize.root_scalar(fn_g, bracket = [gx0, gx1], maxiter = 100)
            #print(r)
            dct_logs['x0_converged'] = r.converged
            if not r.converged:
                dct_logs['found'] = False
                return None, dct_logs
                #raise RuntimeError('H has at least one negative eigenvalue, D is singular, and no solution was found to regularize H.')
            
            x0 = r.root

        dct_logs['x0'] = x0
        dct_logs['f(x0)'] = f(x0)
        if dct_logs['f(x0)'] <= 0:
            dct_logs['found'] = False
            return None, dct_logs
        """
        if f(x0) <= 0:
            print(' === Diagnostics: f(x0) <= 0 ; f({}) = {} ==='.format(x0, f(x0)))
            print('float64 = {}'.format(do_float64))
            print('H eigs = {}'.format(torch.linalg.eigh(H).eigenvalues))
            print('D_inv @ H @ D_inv eigs = {}'.format(torch.linalg.eigh(D_inv @ H @ D_inv).eigenvalues))

            H = H.to(dtype = torch.float64)
            D_inv = D_inv.to(dtype = torch.float64)
            print('f64: H eigs = {}'.format(torch.linalg.eigh(H).eigenvalues))
            print('f64: D_inv @ H @ D_inv eigs = {}'.format(torch.linalg.eigh(D_inv @ H @ D_inv).eigenvalues))
        """
        
        #if f(x0) <= 0:
        #    print(' === Diagnostics (float64 = {}): f(x0) <= 0 ; f({}) = {} ==='.format(do_float64, x0, f(x0)))
        """
        n_reducs = 0
        while f(x0) <= 0:
            #print('reducing x0: x0 = {}, f(x0) = {}'.format(x0, f(x0)))
            x0 *= .9
            n_reducs += 1
        if n_reducs > 0:
            print('n_reducs = {}'.format(n_reducs))
        #print('x0 final:', x0)
        """

    # Compute x1
    x1 = x0 + 1.
    i = 0
    #print('f(x0) = {}, f(x1) = {}'.format(f(x0), f(x1)))
    while f(x1) >= 0:
        #print('increasing x1: x1 = {}, f(x1) = {}'.format(x1, f(x1)))
        x1 *= 3
        i += 1
        if i >= 40:
            raise RuntimeError('Impossible to find a solution to Nesterov problem.')
    dct_logs['x1'] = x1
    dct_logs['f(x1)'] = f(x1)
    #print('final x1: x1 = {}, f(x1) = {}'.format(x1, f(x1)))

    # Compute lrs
    r = scipy.optimize.root_scalar(f, bracket = [x0, x1], maxiter = 100) #, rtol = 1e-4)
    dct_logs['r'] = r.root
    dct_logs['r_converged'] = r.converged
    dct_logs['found'] = r.converged
    return compute_lrs(r.root).to(dtype = dtype), dct_logs