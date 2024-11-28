"""
usage:

from common import *
"""
from common.imports import *

# from expression import Some, pipe, Ok, Result
# from expression import compose, identity
# from expression.collections import seq, Seq

T = TypeVar('T')
V = TypeVar('V')

def subdict(d, keys):
    return {k: d[k] for k in keys if k in d}

def collect_from_batch(batch, key, return_tensors="pt"):
    """
    equivalent to [sample.key for sample in batch], but with additional torch.stack calls
    """
    assert len(batch) > 0
    assert all(hasattr(sample, key) for sample in batch)

    collected = [getattr(sample, key) for sample in batch]
    if isinstance(getattr(batch[0], key), torch.Tensor):
        names = collected[0].names
        collected = torch.stack([sample.rename(None) for sample in collected])
        if return_tensors=="np":
            collected = collected.cpu().numpy()
    elif isinstance(getattr(batch[0], key), np.ndarray):
        names = None
        collected = np.stack([getattr(sample, key) for sample in batch])
        if return_tensors=="pt":
            collected = torch.from_numpy(collected)

    if isinstance(collected, torch.Tensor):
        if names and not all(n is None for n in names):
            collected = collected.refine_names("batch", *names)
    return collected

# region robust code

def not_none(value: T | None) -> T:
    assert value is not None, "value is None"
    return value

# endregion

from common.named_tensor import *

from common.plt import PlotImage, plot_images

def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    else:
        return torch.tensor(x)


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

def dot(a, b, dim=-1, keepdims=True):
    """
    a: torch.Tensor, [B, ...]
    b: torch.Tensor, [B, ...]

    return: torch.Tensor, [B]
    """
    return (a * b).sum(dim=dim, keepdim=keepdims)

def sobel_gradient(image: torch.Tensor):
    """
    image: channel, height, width
    """
    image = image.align_to("channel", "height", "width").rename(None)
    image = image.unsqueeze(0) # b=1 c h w
    sobel_x = torch.tensor([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]).to(image).view(1, 1, 3, 3).repeat(image.shape[1], 1, 1, 1)
    sobel_y = torch.tensor([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ]).to(image).view(1, 1, 3, 3).repeat(image.shape[1], 1, 1, 1)

    grad_x = F.conv2d(image, sobel_x, padding=1, groups=image.shape[1])
    grad_y = F.conv2d(image, sobel_y, padding=1, groups=image.shape[1])
    grad_magnitude = torch.sqrt((grad_x**2 + grad_y**2).sum(dim=1) + 1e-4) # b h w

    return grad_magnitude.squeeze(0).refine_names("height", "width") # h w

def bgr_to_rgb(image: torch.Tensor):
    assert "bgr" in image.names
    image = image.align_to(..., "bgr")
    names = image.names

    assert image.size("bgr") >= 3
    image = image.rename(None)[..., [2, 1, 0]].refine_names(*names[:-1], "rgb")
    return image

def vector_to_point(v: torch.Tensor):
    """
    v: torch.Tensor, [..., 3]

    return: torch.Tensor, [..., 4]
    """
    assert v.size(-1) == 3
    return torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)

def point_to_vector(p: torch.Tensor):
    """
    p: torch.Tensor, [..., 4]

    return: torch.Tensor, [..., 3]
    """
    assert p.size(-1) == 4
    return p[..., :3]

def all_equal(x: List[T]) -> bool:
    return len(x) == 0 or all(x[0] == e for e in x)

from common.random import *
from common.gpu import *
from common.io import *
