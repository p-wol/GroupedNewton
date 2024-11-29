import copy
import pytest
import torch
import grnewt
from grnewt import ParamGroups

def _build_tensors_from_shapes(lst_shapes):
    return [{"params": [torch.randn(*shape) for shape in group_shapes]} for group_shapes in lst_shapes]

@pytest.mark.parametrize("lst_shapes", [
    [[[3, 4], [6], [1, 2, 3]]],
    [[[3, 4]], [[6]], [[1, 2, 3]]],
    [[[3, 4], [6]], [[6]], [[1, 2, 3], [7], [2, 3]]]
    ])
def test_dot_full(lst_shapes):
    pgroups1 = _build_tensors_from_shapes(lst_shapes)
    param_groups1 = ParamGroups(pgroups1)
    pgroups2 = _build_tensors_from_shapes(lst_shapes)
    param_groups2 = ParamGroups(pgroups2)

    res_custom = param_groups1.dot(param_groups1.tup_params, param_groups2.tup_params)
    res_pytorch = torch.tensor([sum([(p1 * p2).sum() for p1, p2 in zip(group1["params"], group2["params"])]) \
            for group1, group2 in zip(pgroups1, pgroups2)])

    assert torch.allclose(res_custom, res_pytorch)

@pytest.mark.parametrize("lst_shapes,start,end", [
    ([[[3, 4], [6], [1, 2, 3]]], 0, None),
    ([[[3, 4], [6], [1, 2, 3]]], 0, 1),
    ([[[3, 4]], [[6]], [[1, 2, 3]]], 0, None),
    ([[[3, 4]], [[6]], [[1, 2, 3]]], 0, 3),
    ([[[3, 4]], [[6]], [[1, 2, 3]]], 1, None),
    ([[[3, 4]], [[6]], [[1, 2, 3]]], 2, None),
    ([[[3, 4], [6]], [[6]], [[1, 2, 3], [7], [2, 3]], [[2, 2], [3, 3]]], 0, None),
    ([[[3, 4], [6]], [[6]], [[1, 2, 3], [7], [2, 3]], [[2, 2], [3, 3]]], 0, 4),
    ([[[3, 4], [6]], [[6]], [[1, 2, 3], [7], [2, 3]], [[2, 2], [3, 3]]], 1, None),
    ([[[3, 4], [6]], [[6]], [[1, 2, 3], [7], [2, 3]], [[2, 2], [3, 3]]], 2, None),
    ([[[3, 4], [6]], [[6]], [[1, 2, 3], [7], [2, 3]], [[2, 2], [3, 3]]], 0, 1),
    ([[[3, 4], [6]], [[6]], [[1, 2, 3], [7], [2, 3]], [[2, 2], [3, 3]]], 1, 3),
    ])
def test_select(lst_shapes, start, end):
    pgroups = _build_tensors_from_shapes(lst_shapes)
    param_groups = ParamGroups(pgroups)

    res_custom = param_groups.select_params(src = param_groups.tup_params, start = start, end = end)

    end2 = end if end is not None else len(lst_shapes)
    res_pytorch = tuple(p for group in pgroups[start:end2] for p in group["params"])
    for p1, p2 in zip(res_custom, res_pytorch):
        assert torch.allclose(p1, p2)

@pytest.mark.parametrize("lst_shapes,start,end", [
    ([[[3, 4], [6], [1, 2, 3]]], 0, None),
    ([[[3, 4], [6], [1, 2, 3]]], 0, 1),
    ([[[3, 4]], [[6]], [[1, 2, 3]]], 0, None),
    ([[[3, 4]], [[6]], [[1, 2, 3]]], 0, 3),
    ([[[3, 4]], [[6]], [[1, 2, 3]]], 1, None),
    ([[[3, 4]], [[6]], [[1, 2, 3]]], 2, None),
    ([[[3, 4], [6]], [[6]], [[1, 2, 3], [7], [2, 3]], [[2, 2], [3, 3]]], 0, None),
    ([[[3, 4], [6]], [[6]], [[1, 2, 3], [7], [2, 3]], [[2, 2], [3, 3]]], 0, 4),
    ([[[3, 4], [6]], [[6]], [[1, 2, 3], [7], [2, 3]], [[2, 2], [3, 3]]], 1, None),
    ([[[3, 4], [6]], [[6]], [[1, 2, 3], [7], [2, 3]], [[2, 2], [3, 3]]], 2, None),
    ([[[3, 4], [6]], [[6]], [[1, 2, 3], [7], [2, 3]], [[2, 2], [3, 3]]], 0, 1),
    ([[[3, 4], [6]], [[6]], [[1, 2, 3], [7], [2, 3]], [[2, 2], [3, 3]]], 1, 3),
    ])
def test_dot_partial(lst_shapes, start, end):
    pgroups1 = _build_tensors_from_shapes(lst_shapes)
    param_groups1 = ParamGroups(pgroups1)
    pgroups2 = _build_tensors_from_shapes(lst_shapes)
    param_groups2 = ParamGroups(pgroups2)

    res_custom = param_groups1.dot(param_groups1.select_params(start = start, end = end), 
            param_groups2.select_params(start = start, end = end), start = start, end = end)

    end2 = end if end is not None else len(lst_shapes)
    lst_res_pytorch = []
    for group1, group2 in zip(pgroups1[start:end2], pgroups2[start:end2]):
        lst_res_pytorch.append(sum([(p1 * p2).sum() for p1, p2 in zip(group1["params"], group2["params"])]))
    res_pytorch = torch.tensor(lst_res_pytorch)

    assert torch.allclose(res_custom, res_pytorch)

#def test_dercon():
#
