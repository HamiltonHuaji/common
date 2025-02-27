"""
usage:

from common import *
"""
from common.imports import *

# from expression import Some, pipe, Ok, Result
# from expression import compose, identity
# from expression.collections import seq, Seq

from gsplat import rasterization
def test_gsplat():
    with torch.device('cuda'):
        _, _, _ = rasterization(
            means=torch.randn(114514, 3),
            quats=torch.randn(114514, 4),
            scales=torch.randn(114514, 3),
            opacities=torch.randn(114514),
            colors=torch.randn(114514, 3),
            viewmats=torch.eye(4)[None],
            Ks=torch.eye(3)[None],
            width=1920,
            height=1080,
        )

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

from common.tensor import *

from common.named_tensor import *

from common.plt import PlotImage, plot_images, restore_mpl_for_jupyter, test_plt

def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    elif isinstance(x, dict):
        return {k: to_tensor(v) for k, v in x.items()}
    elif isinstance(x, list):
        return [to_tensor(v) for v in x]
    elif isinstance(x, tuple):
        return tuple(to_tensor(v) for v in x)
    elif isinstance(x, (float, int, bool)):
        return torch.tensor(x)
    else:
        return torch.tensor(x)

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
from common.threedv import *

def make_laplacian(num_channels=3):
    laplacian_kernel = torch.tensor([
        [0,  1, 0],
        [1, -4, 1],
        [0,  1, 0]
    ], dtype=torch.float32)[None, None].repeat(num_channels, num_channels, 1, 1)
    laplacian = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1, padding_mode='replicate', bias=False)
    laplacian.weight.data = laplacian_kernel
    return laplacian

@torch.inference_mode()
def init_sam2_state_from_video_tensor(
    self,
    video, # b c h w, [0, 1]
    dim_indexing='b h w c',
    offload_video_to_cpu=False,
    offload_state_to_cpu=False,
    async_loading_frames=False,

    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
):
    compute_device = self.device  # device of the model
    video_frames, _, video_height, video_width = video.shape
    with torch.device(compute_device):
        img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
        img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
        images = (rearrange(video, f"{dim_indexing} -> b c h w") - img_mean) / img_std
    inference_state = {}
    inference_state["images"] = images
    inference_state["num_frames"] = len(images)
    # whether to offload the video frames to CPU memory
    # turning on this option saves the GPU memory with only a very small overhead
    inference_state["offload_video_to_cpu"] = offload_video_to_cpu
    # whether to offload the inference state to CPU memory
    # turning on this option saves the GPU memory at the cost of a lower tracking fps
    # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
    # and from 24 to 21 when tracking two objects)
    inference_state["offload_state_to_cpu"] = offload_state_to_cpu
    # the original video height and width, used for resizing final output scores
    inference_state["video_height"] = video_height
    inference_state["video_width"] = video_width
    inference_state["device"] = compute_device
    if offload_state_to_cpu:
        inference_state["storage_device"] = torch.device("cpu")
    else:
        inference_state["storage_device"] = compute_device
    # inputs on each frame
    inference_state["point_inputs_per_obj"] = {}
    inference_state["mask_inputs_per_obj"] = {}
    # visual features on a small number of recently visited frames for quick interactions
    inference_state["cached_features"] = {}
    # values that don't change across frames (so we only need to hold one copy of them)
    inference_state["constants"] = {}
    # mapping between client-side object id and model-side object index
    inference_state["obj_id_to_idx"] = OrderedDict()
    inference_state["obj_idx_to_id"] = OrderedDict()
    inference_state["obj_ids"] = []
    # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
    inference_state["output_dict_per_obj"] = {}
    # A temporary storage to hold new outputs when user interact with a frame
    # to add clicks or mask (it's merged into "output_dict" before propagation starts)
    inference_state["temp_output_dict_per_obj"] = {}
    # Frames that already holds consolidated outputs from click or mask inputs
    # (we directly use their consolidated outputs during tracking)
    # metadata for each tracking frame (e.g. which direction it's tracked)
    inference_state["frames_tracked_per_obj"] = {}
    # Warm up the visual backbone and cache the image feature on frame 0
    self._get_image_feature(inference_state, frame_idx=0, batch_size=1)
    return inference_state
