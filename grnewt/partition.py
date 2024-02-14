from typing import List, Dict, Any, Optional
import torch

def check(key: str, name: str):
    """
    Check if the name of a parameter ('key') matches a given 'name'.
    Typically, name == 'weight' or name == 'bias'.
    """
    i = key.rfind(name)
    if i == -1:
        return False
    elif i + len(name) != len(key):
        return False
    else:
        if i == 0:
            return True
        elif key[i-1] == '.':
            return True
        else:
            return False

def canonical(model: torch.nn.Module):
    """
    Build a partition based on the tensors of parameters of the model.
    """
    return [{'params': [p]} for p in model.parameters()], \
            [[n] for n, p in model.named_parameters()]

def trivial(model: torch.nn.Module):
    """
    Build a partition grouping all the parameters of the model.
    """
    return [{'params': [p for p in model.parameters()]}], \
            [[n for n, p in model.named_parameters()]]

def names(model: torch.nn.Module, lst_names: List[str]):
    """
    Build a partition containing len(lst_names) + 1 groups of parameters.
    The parameters are grouped based on the end of their names.
    Example: 
        With model = torch.nn.Linear(2, 3) and lst_names = ['weight', 'bias'],
        the partition will be: [{'params': [model.weight]}, {'params': [model.bias]}, {'params': []}].
        The last group gathers all the parameters that do not match any of the names in lst_names.
    """
    # Build the groups
    nb_names = len(lst_names)
    param_groups = [{'params': []} for i in range(nb_names + 1)]
    name_groups = [[] for i in range(nb_names + 1)]
    for k, v in model.named_parameters():
        found = False
        for i, name in enumerate(lst_names):
            if check(k, 'weight'):
                param_groups[i]['params'].append(v)
                name_groups[i].append(k)
                found = True
                break

        if not found:
            param_groups[nb_names]['params'].append(v)
            name_groups[nb_names].append(k)

    return param_groups, name_groups

def wb(model: torch.nn.Module):
    """
    Build a partition containing 3 groups of parameters: weights, biases and other.
    """
    return names(model, ['weight', 'bias'])

def blocks(model: torch.nn.Module, num_blocks):
    params = list(model.named_parameters())
    num_params = len(params)

    lst_nums = []
    p = num_params
    b = num_blocks
    while b > 0:
        if p % b == 0:
            lst_nums += [p // b for i in range(b)]
            break
        else:
            lst_nums.append(p // b + 1)
            p -= p // b + 1
            b -= 1

    param_groups = [{'params': []} for i in range(num_blocks * 2)]
    name_groups = [[] for i in range(num_blocks * 2)]
    i_block = 0
    for k, v in params:
        if check(k, 'weight'):
            param_groups[i_block * 2]['params'].append(v)
            name_groups[i_block * 2].append(k)
        elif check(k, 'bias'):
            param_groups[i_block * 2 + 1]['params'].append(v)
            name_groups[i_block * 2 + 1].append(k)
        else:
            raise NotImplementedError('Cannot handle other parameters than "weight" or "bias".')

        lst_nums[i_block] -= 1
        if lst_nums[i_block] == 0:
            i_block += 1

    return param_groups, name_groups
