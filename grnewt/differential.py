import warnings
from itertools import product, repeat, permutations, combinations, combinations_with_replacement
import torch
from .folding import tensor_fold, tensor_unfold

# TODO: add a function to check the precision of the computation of the derivatives
#       e.g.: compute all the elements of K independently, then compare the elements which should theoretically be identical

def taylor_n(f, tup_params, N, lst_v, detach = True):
    """
    Compute the order-N Taylor term of f according to the parameters 'tup_params', in the directions
    given by the list of vectors 'lst_v'.

    n-th "Taylor term": (d^n f)/(d t^n)[v1, v2, ..., vn]
    It does not include the factor 1/n!

    Arguments:
        f: scalar to differentiate
        tup_params: tuple containing the parameters according to which f will be differentiated
        N: order of the derivation
        lst_v: list of vectors on which the differentials will be applied
        detach: if True, detach the output

    Let f0, g, H be respectively the value of f, its gradient and its Hessian at a given point.
    Example:
        We want to compute [f0, v1^T g, v1^T H v2]. We have to call:
        'taylor_n(2, tup_params, f, lst_v = [v1, v2])'
    """
    ### Arguments check
    if len(lst_v) != N:
        raise ValueError(f'len(lst_v) shoud be equal to N = {N}. Found {len(lst_v)} instead.')

    ### Define useful variables
    if tup_params[0].dtype != torch.float64:
        s = 'The recommended floating-point type is torch.float64; found {}. '.format(tup_params[0].dtype) \
                + 'Computations of the derivatives may be imprecise or wrong.'
        warnings.warn(s, UserWarning)
    device, dtype = tup_params[0].device, tup_params[0].dtype

    ### Function: option_detach
    def option_detach(t):
        return t.detach() if detach else t

    ### Compute the Taylor terms
    # Initialize output object "diff_n"
    diff_n = torch.empty(N + 1, device = device, dtype = dtype)

    # Procedure
    deriv = f
    diff_n[0] = option_detach(deriv)
    for n, v in zip(range(1, N + 1), lst_v):
        deriv = torch.autograd.grad(deriv, tup_params, None, create_graph = True)
        deriv = sum([(u_ * v_).sum() for u_, v_ in zip(deriv, v)])
        diff_n[n] = option_detach(deriv)

    return diff_n

def diff_1(f, tup_params, detach = True):
    """
    Compute the order-1 differential of f according to the parameters 'tup_params'.

    Arguments:
        f: scalar to differentiate
        tup_params: tuple containing the parameters according to which f will be differentiated
        detach: if True, detach the output
    """
    deriv = torch.autograd.grad(f, tup_params, None, create_graph = True)

    if detach:
        return tuple(t.detach() for t in deriv)
    else:
        return deriv

def pearlmutter_n(f, tup_params, N, lst_v, output = 'all', detach = True):
    """
    Compute the order-N differential of f according to the parameters 'tup_params', applied to
    the list of N-1 vectors 'lst_v'. The result is a vector.
    This is the generalized version of the "Pearlmutter trick" (or "Hessian-vector product").

    Arguments:
        f: scalar to differentiate
        tup_params: tuple containing the parameters according to which f will be differentiated
        N: order of the differentiation
        lst_v: list of vectors on which the differentials will be applied
        output: either 'all' or 'last': vectors to output (all of the last one only)
        detach: if True, detach the output

    Let H be the Hessian of f.
    Example:
        We want to compute H v1. We have to call:
        'diff_orderN(2, tup_params, f, lst_v = [v1])'
    """
    ### Arguments check
    if output not in ['all', 'last']:
        raise ValueError('"output" shoud be either "all" or "last".')
    if len(lst_v) != N - 1:
        raise ValueError(f'len(lst_v) shoud be equal to N - 1 = {N-1}. Found {len(lst_v)} instead.')

    ### Define useful variables
    if tup_params[0].dtype != torch.float64:
        s = 'The recommended floating-point type is torch.float64; found {}. '.format(tup_params[0].dtype) \
                + 'Computations of the derivatives may be imprecise or wrong.'
        warnings.warn(s, UserWarning)
    device, dtype = tup_params[0].device, tup_params[0].dtype

    ### Function: option_detach
    def option_detach(t):
        if isinstance(t, tuple):
            return tuple(t_.detach() if detach else t_ for t_ in t)
        elif torch.is_tensor(t):
            return t.detach() if detach else t
        else:
            raise TypeError('option_detach: "t" should be either torch.Tensor or tuple.')

    ### Compute the derivatives
    # Initialize output object "diff_n"
    if output == 'all':
        diff_n = [None] * N
    elif output == 'last':
        pass
    else:
        raise NotImplementedError('output = {}'.format(output))

    # Procedure
    deriv = torch.autograd.grad(f, tup_params, None, create_graph = True)

    if output == 'all':
        diff_n[0] = option_detach(deriv)

    for n, v in zip(range(1, N), lst_v):
        deriv = sum([(u_ * v_).sum() for u_, v_ in zip(deriv, v)])
        deriv = torch.autograd.grad(deriv, tup_params, None, create_graph = True)

        if output == 'all':
            diff_n[n] = option_detach(deriv)

    if output == 'all':
        return diff_n
    else:
        return option_detach(deriv)

def features_n(f, tup_params, N, v, use_symmetries = True):
    """
    Compute the order-N tensor summarizing the N-th differential of f in the direction of v.

    Arguments:
        f: scalar to differentiate
        tup_params: tuple containing the parameters according to which f will be differentiated
        N: order of the differentiation
        v: vector on which the differentials will be applied
        use_symmetries: if True, complete K by using its symmetries
    """
    ### Define useful variables
    if tup_params[0].dtype != torch.float64:
        warnings.warn(f'The recommended floating-point type is torch.float64; found {tup_params[0].dtype}. ' \
                + 'Computations of the derivatives may be imprecise or wrong.', UserWarning)
    device, dtype = tup_params[0].device, tup_params[0].dtype

    def compute_Kelem(lst_i):
        deriv = f
        vi = None

        for i in lst_i:
            deriv = torch.autograd.grad(deriv, tup_params[i], vi, create_graph = True)[0]
            vi = v[i]

        return (vi * deriv).sum()

    K = torch.empty(*[len(tup_params)] * N, device = device, dtype = dtype)
    if use_symmetries:
        for tup_idx in combinations_with_replacement(range(len(tup_params)), N):
            try:
                K[tup_idx] = compute_Kelem(list(tup_idx)).detach()
            except:
                #print('Warning: autograd failed at {}'.format(tup_idx))
                K[tup_idx] = 0
                
            set_copy = set(permutations(tup_idx, N))
            set_copy.remove(tup_idx)
            for oth_idx in set_copy:
                K[oth_idx] = K[tup_idx]
    else:
        for tup_idx in product(*repeat(range(len(tup_params)), N)):
            try:
                K[tup_idx] = compute_Kelem(list(tup_idx)).detach()
            except:
                #print('Warning: autograd failed at {}'.format(tup_idx))
                K[tup_idx] = 0

    return K

def term_orderN(N, fn_feat, lst_x_tr, lst_y_tr):
    """
    Takes a function computing the n-th derivative, and compute it for all possible N-uples
    of data points sampled in lst_x_tr, lst_y_tr
    """
    iterN = tuple(zip(lst_x_tr, lst_y_tr) for _ in range(N))
    kernel_sum = None
    for lst_xy in product(*iterN):
        kernel = fn_feat(lst_xy)
        if kernel_sum is None:
            kernel_sum = kernel
        else:
            kernel_sum += kernel

    return kernel_sum


def feat_orderN(model, loss, dct_params, x_ts, y_ts, lst_xy):
    """
    lst_xy = [(x_1, y_1), (x_2, y_2), ...]
    loss(y_estimated, y_target)
    dct_parameters: OrderedDict(model.named_parameters())
    (x_ts, y_ts): 'test' data point at which we will compute the N-th derivative
    lst_xy: list of length N of data points at which we will compute the first order derivative
        (used as input of the N-th derivative, which is a N-linear form)

    batch use (batch of ts): 
        x_ts.size() == (batch_size, ...)
        y_ts.size() == (batch_size, ...)
        loss: returns the sum of losses over the batch
    """
    N = len(lst_xy)
    model.zero_grad()

    tup_params = tuple(v for k, v in dct_params.items())
    if tup_params[0].dtype != torch.float64:
        warnings.warn(f'The recommended floating-point type is torch.float64; found {tup_params[0].dtype}. ' \
                + 'Computations of the derivatives may be imprecise or wrong.', UserWarning)

    def compute_Kelem(lst_i):
        deriv = loss(model(x_ts), y_ts)
        jac_tr = None

        for i, (x_tr, y_tr) in zip(lst_i, lst_xy):
            deriv = torch.autograd.grad(deriv, tup_params[i], jac_tr, create_graph = True)[0]
            jac_tr = torch.autograd.grad(loss(model(x_tr), y_tr), tup_params[i],
                                         retain_graph = True)[0].detach()

        return (jac_tr * deriv).sum()

    device = tup_params[0].device
    K = torch.empty(*[len(tup_params)] * N, device = device)
    for tup_idx in product(*repeat(range(len(tup_params)), N)):
        try:
            K[tup_idx] = compute_Kelem(list(tup_idx)).detach()
        except:
            #print('Warning: autograd failed at {}'.format(tup_idx))
            K[tup_idx] = 0

    return K


def feat_orderN_batch_tr(N, model, loss, dct_params, x_ts, y_ts, x_tr, y_tr,
        optimize_with_symmetries = True):
    """
    lst_xy = [(x_1, y_1), (x_2, y_2), ...]
    loss(y_estimated, y_target)
    dct_parameters: OrderedDict(model.named_parameters())
    (x_ts, y_ts): 'test' data point at which we will compute the N-th derivative
    lst_xy: list of length N of data points at which we will compute the first order derivative
        (used as input of the N-th derivative, which is a N-linear form)

    batch use (batch of ts): 
        x_ts.size() == (batch_size, ...)
        y_ts.size() == (batch_size, ...)
        loss: returns the sum of losses over the batch
    """
    model.zero_grad()

    tup_params = tuple(v for k, v in dct_params.items())
    if tup_params[0].dtype != torch.float64:
        warnings.warn(f'The recommended floating-point type is torch.float64; found {tup_params[0].dtype}. ' \
                + 'Computations of the derivatives may be imprecise or wrong.', UserWarning)

    tup_jac_tr = torch.autograd.grad(loss(model(x_tr), y_tr), tup_params, retain_graph = True)
    tup_jac_tr = tuple(j.detach() for j in tup_jac_tr)

    def compute_Kelem(lst_i):
        deriv = loss(model(x_ts), y_ts)
        jac_tr = None

        for i in lst_i:
            deriv = torch.autograd.grad(deriv, tup_params[i], jac_tr, create_graph = True)[0]
            jac_tr = tup_jac_tr[i]

        return (jac_tr * deriv).sum()

    device = tup_params[0].device
    K = torch.empty(*[len(tup_params)] * N, device = device)
    if optimize_with_symmetries:
        for tup_idx in combinations_with_replacement(range(len(tup_params)), N):
            try:
                K[tup_idx] = compute_Kelem(list(tup_idx)).detach()
            except:
                print('Warning: autograd failed at {}'.format(tup_idx))
                K[tup_idx] = 0
                
            set_copy = set(permutations(tup_idx, N))
            set_copy.remove(tup_idx)
            for oth_idx in set_copy:
                K[oth_idx] = K[tup_idx]
    else:
        for tup_idx in product(*repeat(range(len(tup_params)), N)):
            try:
                K[tup_idx] = compute_Kelem(list(tup_idx)).detach()
            except:
                print('Warning: autograd failed at {}'.format(tup_idx))
                K[tup_idx] = 0

    return K

def split_mat(kernel, tup_names):
    lst_weights = torch.tensor([i for i, name in enumerate(tup_names) if 'weight' in name], dtype = torch.int)
    lst_biases = torch.tensor([i for i, name in enumerate(tup_names) if 'bias' in name], dtype = torch.int)

    dim = kernel.dim()
    lst_names = ['w'*(dim-i) + 'b'*i for i in range(dim + 1)]

    dct_k = {}
    for s in lst_names:
        curr_kernel = kernel
        for i, c in enumerate(s):
            if c == 'w':
                lst_select = lst_weights
            elif c == 'b':
                lst_select = lst_biases
            else:
                raise NotImplemented(f'Unknown parameter code: {c}')

            curr_kernel = curr_kernel.index_select(i, lst_select)
        dct_k[s] = curr_kernel

    return dct_k
