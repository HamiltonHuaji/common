"""
All about 3D Vision
"""

from common.imports import *
from jaxtyping import jaxtyped, Float, Bool, Integer, Shaped
from typeguard import typechecked as typechecker

def with_intrinsics_crop(image, intrinsics, top: int, left: int, height: int, width: int, dim_indexing='b c h w'):
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if dim_indexing != 'b c h w':
        image = rearrange(image, f'{dim_indexing} -> b c h w')
    b, c, h, w = image.shape

    image = torchvision.transforms.functional.crop(image, top, left, height, width)
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, 2] -= left
    intrinsics[..., 1, 2] -= top

    return rearrange(image, f'b c h w -> {dim_indexing}'), intrinsics

# @jaxtyped(typechecker=typechecker)
def mask_axis_aligned_bbox(mask: Bool[torch.Tensor, '... h w']) -> Integer[torch.Tensor, '... 4']:
    """
    Get the axis-aligned bounding box of the pixels where the mask is True.

    Args:
        mask: mask of shape (..., h, w)

    Returns:
        axis-aligned bounding box of shape (..., 4), where the last dimension is (y_min, x_min, y_max, x_max)
    """
    # y_min, x_min, y_max, x_max = mask_axis_aligned_bbox(traffic_light_mask.squeeze(0)).tolist()
    # plt.imshow(traffic_light_mask.expand(3, -1, -1).permute(1, 2, 0).float().cpu().numpy())
    # plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, edgecolor='r', facecolor='none'))
    with mask.device:
        mask_squeeze_w = mask.any(dim=-1).to(torch.int8)
        mask_squeeze_h = mask.any(dim=-2).to(torch.int8)
        x_min = mask_squeeze_h.argmax(dim=-1)
        x_max = mask.size(-1) - mask_squeeze_h.flip(dims=(-1,)).argmax(dim=-1) - 1
        y_min = mask_squeeze_w.argmax(dim=-1)
        y_max = mask.size(-2) - mask_squeeze_w.flip(dims=(-1,)).argmax(dim=-1) - 1
        return torch.stack([y_min, x_min, y_max, x_max], dim=-1)

# @jaxtyped(typechecker=typechecker)
def quaternion_to_matrix(wxyz: Float[torch.Tensor, '*batch 4']) -> Float[torch.Tensor, '*batch 3 3']:
    w, x, y, z = torch.unbind(F.normalize(wxyz, p=2, dim=-1), dim=-1)
    return torch.stack([
        1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y),
        2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x),
        2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y),
    ], dim=-1).view(wxyz.shape[:-1] + (3, 3))

# @jaxtyped(typechecker=typechecker)
def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)

# @jaxtyped(typechecker=typechecker)
def matrix_to_quaternion(matrix: Float[torch.Tensor, '*batch 3 3']) -> Float[torch.Tensor, '*batch 4']:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
        """
        Returns torch.sqrt(torch.max(0, x))
        but with a zero subgradient where x is 0.
        """
        ret = torch.zeros_like(x)
        positive_mask = x > 0
        if torch.is_grad_enabled():
            ret[positive_mask] = torch.sqrt(x[positive_mask])
        else:
            ret = torch.where(positive_mask, torch.sqrt(x), ret)
        return ret

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)

def rigid_registration(
    p: Float[torch.Tensor, '*point 3'],
    q: Float[torch.Tensor, '*point 3'],
    w: Optional[Float[torch.Tensor, '*point']] = None,
    eps: float = 1e-12
) -> Tuple[Float[torch.Tensor, ''], Float[torch.Tensor, '3 3'], Float[torch.Tensor, '3']]:

    with p.device:
        q = q.to(p)
        w = w.to(p) if w is not None else torch.ones(p.shape[:-1])

        p = rearrange(p, "... c -> (...) c")
        q = rearrange(q, "... c -> (...) c")
        w = rearrange(w, "... -> (...)")

        centroid_p = (p * w[..., None]).sum(dim=0) / (w.sum() + eps)
        centroid_q = (q * w[..., None]).sum(dim=0) / (w.sum() + eps)

        p_centered = p - centroid_p
        q_centered = q - centroid_q
        w = w / (torch.sum(w) + eps)
        
        cov = (w[:, None] * p_centered).T @ q_centered
        U, S, Vh = torch.linalg.svd(cov)
        R = Vh.T @ U.T
        if torch.linalg.det(R) < 0:
            Vh[2, :] *= -1
            R = Vh.T @ U.T

        scale = torch.sum(S) / torch.trace((w[:, None] * p_centered).T @ p_centered)
        t = centroid_q - scale * (centroid_p @ R.T)

        return scale, R, t


# def rigid_registration_ransac(
#     p: np.ndarray,
#     q: np.ndarray,
#     w: np.ndarray = None,
#     max_iters: int = 20,
#     hypothetical_size: int = 10,
#     inlier_thresh: float = 0.02
# ) -> Tuple[float, np.ndarray, np.ndarray]:
#     n = p.shape[0]
#     if w is None:
#         w = np.ones(p.shape[0])
    
#     best_score, best_inlines = 0., np.zeros(n, dtype=bool)
#     best_solution = (np.array(1.), np.eye(3), np.zeros(3))

#     for _ in range(max_iters):
#         maybe_inliers = np.random.choice(n, size=hypothetical_size, replace=False)
#         try:
#             s, R, t = rigid_registration(p[maybe_inliers], q[maybe_inliers], w[maybe_inliers])
#         except np.linalg.LinAlgError:
#             continue
#         transformed_p = s * p @ R.T + t
#         errors = w * np.linalg.norm(transformed_p - q, axis=1)
#         inliers = errors < inlier_thresh
        
#         score = inlier_thresh * n - np.clip(errors, None, inlier_thresh).sum()
#         if  score > best_score:
#             best_score, best_inlines = score, inliers
#             best_solution = rigid_registration(p[inliers], q[inliers], w[inliers])
    
#     return best_solution, best_inlines

@torch.no_grad()
def run_trellis(
    self,
    image: Float[torch.Tensor, 'b c 518 518'],
    num_samples: int = 1,
    seed: int = 42,
    sparse_structure_sampler_params: dict = {},
    slat_sampler_params: dict = {},
    formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
) -> dict:
    with self.device, torch.cuda.device(self.device.index):
        cond = self.get_cond(image)
        torch.manual_seed(seed)
        coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)

def moge_depth(
    moge,
    image: torch.Tensor, 
    fov_x: Union[Number, torch.Tensor] = None,
    mini_batch_size: int = 8,
    padding_depth: float = None,
    **kwargs
):
    if image.ndim == 3:
        results = moge.infer(image, force_projection, resolution_level, apply_mask, fov_x)
    else:
        masks = []
        depths = []
        batch_size = image.shape[0]
        if isinstance(fov_x, Number):
            fov_x = torch.tensor(fov_x, device=image.device).expand(batch_size)
        for i in range(0, batch_size, mini_batch_size):
            outputs = moge.infer(image=image[i:i+mini_batch_size], fov_x=fov_x[i:i+mini_batch_size], **kwargs)
            masks.append(outputs['mask'])
            depths.append(outputs['depth'])
        results = {
            'mask': torch.cat(masks, dim=0),
            'depth': torch.cat(depths, dim=0),
        }
    if padding_depth is not None:
        results['depth'] = torch.where(results['mask'] > 0.5, results['depth'], padding_depth)
    return results

def raft_flow(raft, *args, **kwargs) -> Float[torch.Tensor, "b 3 h w"]:
    with torch.inference_mode():
        raft_result = raft(*args, **kwargs)
    mask = torch.ones_like(raft_result['final'][:, 0:1, :, :])
    return torch.cat([raft_result['final'], mask], dim=1)

def backward_warp(image: Float[torch.Tensor, "b h w c"], flow: Float[torch.Tensor, "b 3 h w"], dim_indexing='b h w c'):
    """
    Backward warp an image using given flow. The pixel value at (y, x) in the warped image is the value at (y + flow[0][1, y, x], x + flow[1][0, y, x]) in the original image.

    Args:
        image: image to be warped. (b, h, w, c)
        flow: optical flow. (b, 3, h, w)
    """
    if image.ndim == 3:
        image = image.unsqueeze(0)
    image = rearrange(image, f'{dim_indexing} -> b c h w')
    b, c, h, w = image.shape
    
    if flow.ndim == 3:
        flow = flow.unsqueeze(0)
    
    with flow.device:
        dx, dy, mask = torch.unbind(flow, dim=1)
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        y, x = y.float() + .5 + dy, x.float() + .5 + dx
        xy = torch.stack([x, y], dim=-1) / torch.tensor([w, h]) * 2. - 1. # normalize to [-1, 1]

        warped_image = F.grid_sample(image, xy, mode='bilinear', padding_mode='zeros', align_corners=False)

        return rearrange(warped_image, f'b c h w -> {dim_indexing}'), torch.minimum(mask, (xy.abs().max(dim=-1).values <= 1).float())

def compose_flow(flow1, flow2):
    """
    Compose two optical flows. flow2 is the flow from frame 2 to frame 3, and flow1 is the flow from frame 1 to frame 2. The composed flow is the flow from frame 1 to frame 3.

    Args:
        flow1: flow from frame 1 to frame 2. (b, 3, h, w)
        flow2: flow from frame 2 to frame 3. (b, 3, h, w)
    """

    # mask<1 表示 flow1 不可用的区域, 或采样过程中超出边界的区域
    warped_flow, mask = backward_warp(image=flow2, flow=flow1, dim_indexing='b c h w')

    composed_flow = flow1[:, 0:2, ...] + warped_flow[:, 0:2, ...]
    composed_mask = torch.minimum(torch.minimum(flow1[:, 2:3, ...], warped_flow[:, 2:3, ...]), mask[:, None, ...])

    return torch.cat([composed_flow, composed_mask], dim=1)

# def backward_warp(image, dy=None, dx=None, dxy=None, dyx=None, dim_indexing='b h w c', return_dict=False, return_tuple=False):
#     """
#     Backward warp an image using dx and dy. The pixel value at (y, x) in the warped image is the value at (y + dy[y, x], x + dx[y, x]) in the original image.

#     Args:
#         image: image to be warped. (b, h, w, c)
#         dy: delta y, in pixels. (b, h, w)
#         dx: delta x, in pixels. (b, h, w)
#     """

#     if image.ndim == 3:
#         image = image.unsqueeze(0)
#     if (dy is not None) and (dy.ndim == 2):
#         dy = dy.unsqueeze(0)
#     if (dx is not None) and (dx.ndim == 2):
#         dx = dx.unsqueeze(0)
#     if (dxy is not None) and (dxy.ndim == 3):
#         dxy = dxy.unsqueeze(0)
#     if (dyx is not None) and (dyx.ndim == 3):
#         dyx = dyx.unsqueeze(0)

#     if (dx is None) or (dy is None):
#         if dxy is not None:
#             dx, dy = torch.unbind(dxy, dim=1)
#         elif dyx is not None:
#             dy, dx = torch.unbind(dyx, dim=1)
#         else:
#             raise ValueError('Either (dy and dx) or dxy or dyx should be provided.')

#     b, h, w, c = image.shape
#     y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
#     y, x = y.float() + .5 + dy, x.float() + .5 + dx
#     xy = torch.stack([x, y], dim=-1) / torch.tensor([w, h]) * 2. - 1. # normalize to [-1, 1]

#     warped_image = F.grid_sample(rearrange(image, f'{dim_indexing} -> b c h w'), xy, mode='bilinear', padding_mode='zeros', align_corners=False)
#     warp_valid_mask = xy.abs().max(dim=-1).values <= 1

#     if return_tuple:
#         return rearrange(warped_image, f'b c h w -> {dim_indexing}'), warp_valid_mask
#     elif return_dict:
#         return {
#             'warped_image': rearrange(warped_image, f'b c h w -> {dim_indexing}'),
#             'warp_valid_mask': warp_valid_mask,
#         }
#     else:
#         return rearrange(warped_image, f'b c h w -> {dim_indexing}')


def backward_warp_with_refinement(img_i, img_j, img_i_mask, img_j_mask, raft, dim_indexing='b h w c', num_iters=100, raft_batch_size=16, pbar=True):
    """
    带额外校正的 backward warp. 将会把 targe image 使用 backward warp 校正到 base image 的视角下.

    Args:
        img_i: base image. (b, h, w, c)
        img_j: target image. (b, h, w, c)
        raft: RAFT model
        img_i_mask: base image mask, 1 for valid pixels. (b, h, w)
        img_j_mask: target image mask, 1 for valid pixels. (b, h, w)
    """

    with torch.inference_mode():
        _img_i = rearrange(img_i, f'{dim_indexing} -> b c h w').contiguous()
        _img_j = rearrange(img_j, f'{dim_indexing} -> b c h w').contiguous()

        if img_i_mask is None:
            img_i_mask = torch.ones_like(_img_i[:, 0, :, :])
        if img_j_mask is None:
            img_j_mask = torch.ones_like(_img_j[:, 0, :, :])

        # dxy = raft(_img_i.contiguous(), _img_j.contiguous(), iters=20, test_mode=True)['final']
        _dxys = []
        for i in range(0, len(_img_i), raft_batch_size):
            _dxys.append(raft(_img_i[i:i+raft_batch_size], _img_j[i:i+raft_batch_size], iters=20, test_mode=True)['final'])
        dxy = torch.cat(_dxys, dim=0)

    learnable_dxy = nn.Parameter(dxy.clone().detach())
    optimizer = torch.optim.Adam([learnable_dxy], lr=1e-3)

    progress_bar = trange(num_iters, disable=not pbar)
    for _ in progress_bar:
        optimizer.zero_grad()

        dx, dy = torch.unbind(learnable_dxy, dim=1) # b h w, b h w
        with torch.no_grad():
            warp_to_i_mask = backward_warp(img_j_mask.unsqueeze(-1).clone().detach().float(), dy, dx, dim_indexing='b h w c').squeeze(-1)
        warp_to_i = backward_warp(img_j.clone().detach(), dy, dx, dim_indexing=dim_indexing)

        warp_error = torch.where((warp_to_i_mask * img_i_mask.float()) > .5, (warp_to_i - img_i).abs().mean(dim=-1), 0).mean()
        warp_distortion = (learnable_dxy - dxy).pow(2).mean()

        loss = warp_error + warp_distortion * 0.1
        loss.backward()
        optimizer.step()

        progress_bar.set_postfix(warp_error=warp_error.item(), warp_distortion=warp_distortion.item())

    dx, dy = torch.unbind(learnable_dxy, dim=1) # b h w, b h w
    return {
        'dxy': dxy,
        'optimized_dxy': learnable_dxy,
        'warped_image': backward_warp(img_j, dy, dx, dim_indexing=dim_indexing),
    }
