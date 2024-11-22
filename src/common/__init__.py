"""
usage:

from common import *
"""
import os
# from dotenv import load_dotenv
# load_dotenv()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import numpy as np

from einops import *

# frequent used modules
import math
import time
import random
import shutil
import datetime
import functools
import itertools
import importlib
import imageio.v3 as imageio
from copy import deepcopy

# clean code & typing
from typing import *
from pathlib import Path
from omegaconf import OmegaConf
from dataclasses import dataclass, field

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

# data processing & visualization
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# debug
import gc
import ipdb
import inspect
import weakref
import traceback
from ipdb import launch_ipdb_on_exception

from IPython import embed
from IPython.display import clear_output, display

# region tqdm

if ("TELEGRAM_BOT_TOKEN" in os.environ) and ("TELEGRAM_CHAT" in os.environ) and (os.environ.get("https_proxy", "") != ""):
    print("Using tqdm.contrib.telegram")
    from tqdm.contrib.telegram import tqdm, trange
    tqdm = functools.partial(tqdm, dynamic_ncols=True, token=os.environ["TELEGRAM_BOT_TOKEN"], chat_id=os.environ["TELEGRAM_CHAT"], mininterval=10, maxinterval=100)
    trange = functools.partial(trange, dynamic_ncols=True, token=os.environ["TELEGRAM_BOT_TOKEN"], chat_id=os.environ["TELEGRAM_CHAT"], mininterval=10, maxinterval=100)
else:
    from tqdm.auto import tqdm, trange
    tqdm = functools.partial(tqdm, dynamic_ncols=True)
    trange = functools.partial(trange, dynamic_ncols=True)

# endregion

# region robust code

def not_none(value: T | None) -> T:
    assert value is not None, "value is None"
    return value

# endregion

# region plot images

@dataclass(kw_only=True)
class PlotImage:
    image: torch.Tensor | np.ndarray
    title: str | None = None
    cmap: str = 'viridis'
    value_range: Tuple[float, float] = (0, 1) # (0, 1): will be plotted as is

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    @classmethod
    def from_tensor(cls, image: torch.Tensor | np.ndarray, title: str | None = None, cmap: str = 'viridis', value_range: Tuple[float, float] = (0, 1)):
        if isinstance(image, torch.Tensor):
            if ("bgr" in image.names):
                image = image.align_to(..., "height", "width", "bgr").rename(None)
                if image.size(-1) >= 3:
                    image = image[..., [2, 1, 0]]
            elif ("rgb" in image.names):
                image = image.align_to(..., "height", "width", "rgb").rename(None)
        return cls(image=image, title=title, cmap=cmap, value_range=value_range)

def plot_images(images, max_cols=4, axis_size=(4, 3)):
    images = [
        PlotImage.from_dict(image)
        if isinstance(image, dict)
        else (
            PlotImage.from_tensor(image)
            if isinstance(image, (torch.Tensor, np.ndarray))
            else image
        )
        for image in images
    ]

    n = len(images)
    n_rows = int(np.sqrt(n))
    n_cols = (n - 1) // n_rows + 1
    assert n_rows * n_cols >= n
    if n_cols > max_cols:
        n_cols = max_cols
        n_rows = (n - 1) // n_cols + 1

    per_axis_size_x, per_axis_size_y = axis_size
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * per_axis_size_x, n_rows * per_axis_size_y), layout='constrained')
    axes = axes.flatten() if n > 1 else [axes]

    for plot_image, ax in zip(images, axes):
        image = plot_image.image.detach().clone().cpu().numpy() if isinstance(plot_image.image, torch.Tensor) else plot_image.image
        image = image.squeeze()

        if image.dtype != np.float32 and image.dtype != np.int32:
            image = image.astype(np.float32)

        # deal with H,W,3 and 3,H,W
        if image.ndim == 3 and image.shape[0] == 3:
            image = image.transpose(1, 2, 0)

        # deal with value_range
        if plot_image.value_range != (0, 1):
            image = (image - plot_image.value_range[0]) / (plot_image.value_range[1] - plot_image.value_range[0])

        # if 1-channel image, show per axis colorbar
        if image.ndim == 2:
            f = ax.imshow(image, cmap=plot_image.cmap)
            plt.colorbar(f, ax=ax)
        else:
            image = np.clip(image, 0, 1)
            ax.imshow(image)

        if plot_image.title:
            ax.set_title(plot_image.title)
    # plt.tight_layout()

# endregion

def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    else:
        return torch.tensor(x)

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

def pink_noise(*, height, width, channels=3, alpha=1.0):
    """
    size: int, size of noise
    alpha: float, 1.0 is pink noise, 2.0 is brown noise

    return: torch.Tensor, [size]
    """
    # Generate white noise
    samples = torch.randn(height, width, channels)

    # Compute Fourier Transform
    spectrum = torch.fft.fftn(samples, dim=(0, 1))

    # Compute frequencies
    fy = torch.fft.fftfreq(height)[:, None]
    fx = torch.fft.fftfreq(width)[None, :]
    frequency = torch.sqrt(fx ** 2 + fy ** 2)
    frequency = frequency ** alpha

    frequency[0, 0] = 1.0

    spectrum = spectrum / frequency.unsqueeze(-1)
    samples = torch.fft.ifftn(spectrum, dim=(0, 1)).real

    # Normalize
    samples = (samples - samples.min()) / (samples.max() - samples.min())
    return samples

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

# region evil code
def occupy_all(device_idx: int, *, margin_size = 128 * 1024 * 1024):
    import gc
    if isinstance(device_idx, str):
        if device_idx == "cpu":
            return
        device_idx = torch.device(device_idx).index
    elif isinstance(device_idx, torch.device):
        if device_idx.type != "cuda":
            return
        device_idx = device_idx.index
    
    if device_idx is None:
        free_bytes, _ = torch.cuda.mem_get_info()
        device = torch.device("cuda")
    else:
        assert isinstance(device_idx, int), f"device_idx must be int, but got {device_idx}({type(device_idx)})"
        import pynvml
        # Step 1: get the available memory in bytes
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_bytes = info.free
        pynvml.nvmlShutdown()
        device = torch.device(f"cuda:{device_idx}")

    # Step 2: occupy the memory
    
    if free_bytes > margin_size:
        try:
            print(f"Free GBs={free_bytes / 1024 ** 3}")
            x = torch.randn((free_bytes - margin_size) // 4, device=device, dtype=torch.float32)
            time.sleep(.5)
            del x
            gc.collect()
        except Exception as e:
            print(f"Warning: occupy_all failed {e}")
    else:
        print("Warning: not enough memory (less than margin_size)")
    return

# endregion


