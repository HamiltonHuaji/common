import torch
import plyfile
import numpy as np
from plyfile import PlyData, PlyElement
from pathlib import Path
from jaxtyping import *

from common.tensor import split_tensor, expand_to

def store_ply(path, xyz: Float[torch.Tensor, 'n 3'], rgb: Float[torch.Tensor, 'n 3']):
    xyz = xyz.reshape(-1, 3)
    rgb = rgb.reshape(-1, 3)
    if isinstance(xyz, torch.Tensor):
        xyz = xyz.detach().float().cpu().numpy()
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().float().cpu().numpy()

    valid_fp = lambda x: ~np.isnan(x) & ~np.isinf(x)
    valid_mask = valid_fp(xyz).all(axis=-1) & valid_fp(rgb).all(axis=-1) # xyz, rgb must be both valid
    xyz, rgb = xyz[valid_mask], rgb[valid_mask]

    rgb = np.clip(rgb * 255, 0., 255.)
    normals = np.zeros_like(xyz)
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def store_3dgs(path: Path, *, means, quats, scales, opacities, shs, camera_position=None):
    path = Path(path)

    if opacities.ndim == 1:
        opacities = opacities.unsqueeze(-1)
    
    if shs.ndim == 3:
        shs = shs.flatten()

    if camera_position is not None:
        assert camera_position.size() == (3,)
        view_directions = F.normalize(means - camera_position[None, :], dim=-1) # (num_gaussians, 3) - (1, 3) -> (num_gaussians, 3)

        # scales, rotations = self.scales, self.quats
        rotations_mat = quaternion_to_matrix(quats)
        min_scales = torch.argmin(scales, dim=-1)
        indices = torch.arange(min_scales.shape[0])
        normals = rotations_mat[indices, :, min_scales] # (num_gaussians, 3)

        view_dot_normals = einsum(view_directions, normals, "n i, n i -> n")
        normals = expand_to(normals, view_directions)
        view_dependent_normals = torch.where((view_dot_normals > 0)[..., None], -normals, normals) # (num_gaussians, 3)
    else:
        view_dependent_normals = torch.zeros_like(means)

    shs_dc, shs_rest = split_tensor(shs, [1 * 3, None], dim=1)

    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    for i in range(shs_dc.size(1)):
        l.append(f"f_dc_{i}")
    for i in range(shs_rest.size(1)):
        l.append(f"f_rest_{i}")
    l.append("opacity")
    for i in range(scales.size(1)):
        l.append(f"scale_{i}")
    for i in range(quats.size(1)):
        l.append(f"rot_{i}")
    dtype_full = [(attribute, 'f4') for attribute in l]
    elements = np.empty(means.size(0), dtype=dtype_full)

    # print(f"means.shape={means.shape}")
    # print(f"view_dependent_normals.shape={view_dependent_normals.shape}")
    # print(f"shs_dc.shape={shs_dc.shape}")
    # print(f"shs_rest.shape={shs_rest.shape}")
    # print(f"opacities.shape={opacities.shape}")
    # print(f"scales.shape={scales.shape}")
    # print(f"quats.shape={quats.shape}")

    attributes = torch.cat([
        means, view_dependent_normals, shs_dc, shs_rest, opacities, scales, quats
    ], dim=1).detach().cpu().numpy()
    elements[:] = list(map(tuple, attributes))
    elements = PlyElement.describe(elements, 'vertex')
    PlyData([elements]).write(path)
