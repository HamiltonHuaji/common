from common.imports import *

def named_where(cond: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
    assert cond.ndim >= a.ndim, (cond.shape, a.shape)
    assert cond.ndim >= b.ndim, (cond.shape, b.shape)
    a = a.align_as(cond)
    b = b.align_as(cond)
    return torch.where(cond.rename(None), a.rename(None), b.rename(None)).refine_names(*cond.names)

def named_expand(t: torch.Tensor, **kwargs):
    """
    named_expand(torch.zeros(3, 1, names=('a', 'b')), c=2, b=2) # shape=(c=2, a=3, b=2)
    """
    new_dims = [k for k in kwargs if k not in t.names]
    expand_expr = [1] * len(new_dims) + [
        kwargs[n] if n in kwargs else -1
        for i, n in enumerate(t.names)
    ]
    return t.rename(None).expand(*expand_expr).refine_names(*new_dims, *t.names)