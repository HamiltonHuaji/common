import torch
import torch.nn.functional as F
import functools
import math
from einops import einsum, rearrange

def to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    elif isinstance(x, list):
        return [to_device(v, device) for v in x]
    elif isinstance(x, tuple):
        return tuple(to_device(v, device) for v in x)
    elif hasattr(x, "to"):
        x_ = x.to(device)
        if x_ is not None:
            return x_ # not in place
        else:
            return x # in place
    else:
        return x

def tensor_hash(x: torch.Tensor) -> float:
    """
    快速计算浮点张量的哈希值（0-1之间）
    特性：
    1. 对元素值变化敏感（每个元素参与计算）
    2. 对均值和方差变化敏感
    3. 使用确定性随机投影保证一致性
    4. 输出范围[0, 1)
    """
    # 展平张量并生成确定性随机投影
    flat = x.flatten().float()
    gen = torch.Generator(device=x.device).manual_seed(42)  # 固定种子保证一致性
    proj = torch.randn(flat.shape[0], generator=gen, device=x.device)
    
    # 核心计算（并行化优化）
    dot = torch.dot(flat, proj)  # 元素级别敏感
    mean = flat.mean()             # 均值敏感
    var = flat.var(unbiased=False) # 方差敏感
    
    # 非线性混合（使用无理数避免周期性）
    h = (dot * math.pi % 1 + 
         mean * math.e % 1 + 
         var * math.sqrt(2) % 1) % 1
    
    return h

# def current_device():
#     return torch.zeros(()).device

def current_device():
    """
    usage:
    with device:
        ...
        a = torch.zeros(())
        assert device == a.device
        assert device == get_current_device()
    """
    from torch._C import _len_torch_function_stack
    from torch.overrides import TorchFunctionMode, _pop_mode, _push_mode
    from torch.utils._device import DeviceContext

    device = torch.get_default_device()

    mode_stack = []
    for _ in range(_len_torch_function_stack()):
        mode = _pop_mode()
        mode_stack.append(mode)
        if isinstance(mode, DeviceContext):
            device = mode.device
            break
    for mode in reversed(mode_stack):
        _push_mode(mode)
    return device

def split_tensor(x, pattern, dim):
    split_dim_size = x.size(dim)
    is_placeholder = lambda p: p is None or p < 0

    n_placeholder = len([p for p in pattern if is_placeholder(p)])
    assert n_placeholder <= 1, "Only one placeholder is allowed"
    if n_placeholder == 0:
        return torch.split(x, pattern, dim)
    else:
        pattern_sum = sum([p for p in pattern if not is_placeholder(p)])
        assert pattern_sum <= split_dim_size, f"Sum of pattern {pattern} is greater than split_dim_size {split_dim_size}"
        pattern = [p if not is_placeholder(p) else split_dim_size - pattern_sum for p in pattern]
        return torch.split(x, pattern, dim)

def expand_to(a, b):
    """
    a: torch.Tensor, [       ...a ]
    b: torch.Tensor, [ ...b, ...a ]

    return: torch.Tensor, [ ...b, ...a ]
    """
    a_shape = a.shape
    b_shape = b.shape

    # a_shape must be suffix of b_shape
    assert a_shape == b_shape[-len(a_shape):], (a_shape, b_shape)
    diff_shape = b_shape[:-len(a_shape)]
    # print(f"diff_shape={diff_shape}")
    c = a.view(*([1] * len(diff_shape)), *a_shape)
    # print(f"c.shape={c.shape}")
    c = c.expand(*diff_shape, *([-1] * len(a_shape)))
    # print(f"c.shape={c.shape}")
    return c
