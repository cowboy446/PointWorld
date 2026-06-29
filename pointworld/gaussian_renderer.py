# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image


SH_C0 = 0.28209479177387814


def rgb_to_sh0(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - 0.5) / SH_C0


def sh0_to_rgb(sh0: torch.Tensor) -> torch.Tensor:
    return torch.clamp(sh0 * SH_C0 + 0.5, 0.0, 1.0)


def inverse_softplus(value: float) -> float:
    x = torch.tensor(float(value), dtype=torch.float32)
    return float(torch.log(torch.expm1(x.clamp_min(1e-8))).item())


@torch.no_grad()
def save_gaussian_ply(
    gaussians: dict[str, torch.Tensor],
    exists: torch.Tensor,
    output_path: str | os.PathLike[str],
) -> int:
    means = gaussians["means"].detach().float()
    sh0 = gaussians["sh0"].detach().float()
    quats = torch.nn.functional.normalize(gaussians["q"].detach().float(), dim=-1, eps=1e-8)
    scales = gaussians["s"].detach().float()
    opacity = gaussians["o"].detach().float().view(-1, 1)

    valid = (
        exists.detach().bool().view(-1)
        & torch.isfinite(means).all(dim=-1)
        & torch.isfinite(sh0).all(dim=-1)
        & torch.isfinite(quats).all(dim=-1)
        & torch.isfinite(scales).all(dim=-1)
        & torch.isfinite(opacity).view(-1)
        & (scales > 0).all(dim=-1)
    )
    means = means[valid].cpu().numpy()
    sh0 = sh0[valid].cpu().numpy()
    quats = quats[valid].cpu().numpy()
    log_scales = torch.log(scales[valid].clamp_min(1e-12)).cpu().numpy()
    opacity_logits = torch.logit(opacity[valid].clamp(1e-6, 1.0 - 1e-6)).cpu().numpy()

    base_property_names = [
        "x", "y", "z", "nx", "ny", "nz",
        "f_dc_0", "f_dc_1", "f_dc_2",
    ]
    sh_rest_property_names = [f"f_rest_{index}" for index in range(45)]
    property_names = base_property_names + sh_rest_property_names + [
        "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    vertices = np.empty(means.shape[0], dtype=np.dtype([(name, "<f4") for name in property_names]))
    values = np.concatenate(
        [
            means,
            np.zeros_like(means),
            sh0,
            np.zeros((means.shape[0], 45), dtype=np.float32),
            opacity_logits,
            log_scales,
            quats,
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    for index, name in enumerate(property_names):
        vertices[name] = values[:, index]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "ply",
        "format binary_little_endian 1.0",
        "comment PointWorld predicted 3D Gaussians; SH degree 0",
        f"element vertex {vertices.shape[0]}",
    ]
    header.extend(f"property float {name}" for name in property_names)
    header.extend(["end_header", ""])
    with output_path.open("wb") as handle:
        handle.write("\n".join(header).encode("ascii"))
        handle.write(vertices.tobytes())
    return int(vertices.shape[0])


def camera_prefixes(data_dict: dict) -> list[str]:
    prefixes = []
    for key in data_dict.keys():
        match = re.fullmatch(r"(cam\d+)_initial_rgb", str(key))
        if match is not None:
            prefix = match.group(1)
            required = [
                f"{prefix}_initial_rgb",
                f"{prefix}_intrinsic",
                f"{prefix}_extrinsic",
            ]
            if all(req in data_dict for req in required):
                prefixes.append(prefix)
    return sorted(prefixes)


def _as_float_rgb(rgb: torch.Tensor) -> torch.Tensor:
    rgb = rgb.float()
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    return rgb.clamp(0.0, 1.0)


def dssim_loss(rendered: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Compact SSIM variant over a masked image. This mirrors the classic 3DGS
    # L1 + D-SSIM objective while keeping the mask semantics requested here.
    valid = mask.expand_as(rendered)
    if not valid.any():
        return rendered.new_zeros(())
    x = rendered[valid].float()
    y = target[valid].float()
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = x.mean()
    mu_y = y.mean()
    var_x = (x - mu_x).pow(2).mean()
    var_y = (y - mu_y).pow(2).mean()
    cov_xy = ((x - mu_x) * (y - mu_y)).mean()
    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / (
        (mu_x.pow(2) + mu_y.pow(2) + c1) * (var_x + var_y + c2)
    )
    return (1.0 - ssim.clamp(-1.0, 1.0)) * 0.5


def _project_points(points: torch.Tensor, intrinsic: torch.Tensor, extrinsic: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ones = torch.ones((points.shape[0], 1), device=points.device, dtype=points.dtype)
    pts_h = torch.cat([points, ones], dim=-1)
    pts_cam = (extrinsic.float() @ pts_h.float().T).T[:, :3]
    pix_h = (intrinsic.float() @ pts_cam.T).T
    z = pix_h[:, 2].clamp_min(1e-6)
    pixels = pix_h[:, :2] / z.unsqueeze(-1)
    return pixels, pts_cam[:, 2]


def _make_projection_matrix(
    intrinsic: torch.Tensor,
    width: int,
    height: int,
    znear: float,
    zfar: float,
) -> torch.Tensor:
    fx = intrinsic[0, 0].float()
    fy = intrinsic[1, 1].float()
    cx = intrinsic[0, 2].float()
    cy = intrinsic[1, 2].float()
    proj = torch.zeros((4, 4), device=intrinsic.device, dtype=torch.float32)
    proj[0, 0] = 2.0 * fx / float(width)
    proj[1, 1] = 2.0 * fy / float(height)
    proj[0, 2] = 2.0 * cx / float(width) - 1.0
    proj[1, 2] = 2.0 * cy / float(height) - 1.0
    proj[3, 2] = 1.0
    proj[2, 2] = float(zfar) / (float(zfar) - float(znear))
    proj[2, 3] = -(float(zfar) * float(znear)) / (float(zfar) - float(znear))
    return proj.t().contiguous()


def _camera_center_from_world_to_camera(extrinsic: torch.Tensor) -> torch.Tensor:
    cam_to_world = torch.linalg.inv(extrinsic.float())
    return cam_to_world[:3, 3].contiguous()


def _projection_mask_from_points(
    points: torch.Tensor,
    intrinsic: torch.Tensor,
    extrinsic: torch.Tensor,
    height: int,
    width: int,
    exists: torch.Tensor,
    radius: int = 2,
) -> torch.Tensor:
    with torch.no_grad():
        pixels, depth = _project_points(points.detach(), intrinsic.detach(), extrinsic.detach())
        valid = (
            exists.bool()
            & torch.isfinite(pixels).all(dim=-1)
            & (depth > 0)
        )
        if not valid.any():
            return torch.zeros((1, height, width), device=points.device, dtype=torch.bool)

        xy = torch.round(pixels[valid]).long()
        x = xy[:, 0]
        y = xy[:, 1]
        in_bounds = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        if not in_bounds.any():
            return torch.zeros((1, height, width), device=points.device, dtype=torch.bool)

        radius = max(int(radius), 0)
        if radius > 0:
            offsets = torch.arange(-radius, radius + 1, device=points.device, dtype=torch.long)
            oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
            x = x[in_bounds, None] + ox.reshape(1, -1)
            y = y[in_bounds, None] + oy.reshape(1, -1)
            in_bounds = (x >= 0) & (x < width) & (y >= 0) & (y < height)
            flat = (y.clamp(0, height - 1) * width + x.clamp(0, width - 1))[in_bounds]
        else:
            flat = y[in_bounds] * width + x[in_bounds]
        mask_flat = torch.zeros((height * width,), device=points.device, dtype=torch.bool)
        mask_flat[flat] = True
        return mask_flat.view(1, height, width)


def _empty_render(device: torch.device, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    image = torch.zeros((3, height, width), device=device, dtype=torch.float32)
    mask = torch.zeros((1, height, width), device=device, dtype=torch.bool)
    return image, mask


def render_single_view_diff_gaussian(
    means: torch.Tensor,
    sh0: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacity: torch.Tensor,
    intrinsic: torch.Tensor,
    extrinsic: torch.Tensor,
    height: int,
    width: int,
    exists: torch.Tensor | None = None,
    mask_points: torch.Tensor | None = None,
    patch_radius: int = 2,
    mask_radius: int = 2,
    znear: float = 0.01,
    zfar: float = 100.0,
    min_render_depth: float = 0.05,
    max_screen_radius: float = 64.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    except Exception as exc:
        raise RuntimeError(
            "diff_gaussian renderer requested but diff_gaussian_rasterization is not importable. "
            "Install third_party/diff-gaussian-rasterization in the pointwm environment."
        ) from exc

    device = means.device
    if exists is None:
        exists = torch.ones(means.shape[0], dtype=torch.bool, device=device)
    pixels, depth = _project_points(means.detach().float(), intrinsic.detach().float(), extrinsic.detach().float())
    min_depth = max(float(znear), float(min_render_depth))
    input_finite = (
        torch.isfinite(means).all(dim=-1)
        & torch.isfinite(sh0).all(dim=-1)
        & torch.isfinite(quats).all(dim=-1)
        & torch.isfinite(scales).all(dim=-1)
        & torch.isfinite(opacity).view(-1)
    )
    valid = (
        exists.bool()
        & input_finite
        & torch.isfinite(pixels).all(dim=-1)
        & torch.isfinite(depth)
        & (depth > min_depth)
        & (depth < float(zfar))
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] < float(width))
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] < float(height))
    )
    if not valid.any():
        return _empty_render(device, height, width)

    # The diff-gaussian CUDA extension expects fp32 tensors and is not autocast-safe.
    with torch.autocast("cuda", enabled=False):
        means_f = means.float()
        intrinsic_f = intrinsic.float()
        extrinsic_f = extrinsic.float()

        means_v = means_f[valid].contiguous()
        colors_v = sh0_to_rgb(sh0[valid].float()).contiguous()
        scales_v = scales[valid].float()
        quats_v = quats[valid].float()
        opacity_v = opacity[valid].float().view(-1, 1)
        if max_screen_radius > 0:
            focal = 0.5 * (intrinsic_f[0, 0].abs() + intrinsic_f[1, 1].abs()).clamp_min(1e-6)
            depth_v = depth[valid].to(device=device, dtype=torch.float32).clamp_min(float(min_depth))
            max_scale_v = (float(max_screen_radius) * depth_v / focal).view(-1, 1)
            scales_v = torch.minimum(scales_v, max_scale_v)
        scales_v = scales_v.clamp_min(1e-7)
        params_finite = (
            torch.isfinite(means_v).all(dim=-1)
            & torch.isfinite(colors_v).all(dim=-1)
            & torch.isfinite(scales_v).all(dim=-1)
            & torch.isfinite(quats_v).all(dim=-1)
            & torch.isfinite(opacity_v).view(-1)
        )
        if not params_finite.all():
            means_v = means_v[params_finite].contiguous()
            colors_v = colors_v[params_finite].contiguous()
            scales_v = scales_v[params_finite]
            quats_v = quats_v[params_finite].contiguous()
            opacity_v = opacity_v[params_finite].contiguous()
            if means_v.numel() == 0:
                return _empty_render(device, height, width)
        scales_v = scales_v.contiguous()
        quats_v = quats_v.contiguous()
        opacity_v = opacity_v.contiguous()
        means2d = torch.zeros_like(means_v, requires_grad=True)

        fx = intrinsic_f[0, 0].clamp_min(1e-6)
        fy = intrinsic_f[1, 1].clamp_min(1e-6)
        tanfovx = float(width) / (2.0 * fx)
        tanfovy = float(height) / (2.0 * fy)
        viewmatrix = extrinsic_f.t().contiguous()
        projmatrix = (viewmatrix @ _make_projection_matrix(intrinsic_f, width, height, znear, zfar)).contiguous()
        campos = _camera_center_from_world_to_camera(extrinsic_f)

        settings = GaussianRasterizationSettings(
            image_height=int(height),
            image_width=int(width),
            tanfovx=float(tanfovx.item()),
            tanfovy=float(tanfovy.item()),
            bg=torch.zeros((3,), device=device, dtype=torch.float32),
            scale_modifier=1.0,
            viewmatrix=viewmatrix,
            projmatrix=projmatrix,
            sh_degree=0,
            campos=campos,
            prefiltered=False,
            debug=False,
        )
        rasterizer = GaussianRasterizer(settings)
        try:
            image, radii_v = rasterizer(
                means3D=means_v,
                means2D=means2d,
                colors_precomp=colors_v,
                opacities=opacity_v,
                scales=scales_v,
                rotations=quats_v,
            )
        except torch.OutOfMemoryError:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(
                "Warning: diff-gaussian rasterizer OOM for one view; falling back to torch renderer "
                f"(valid_gaussians={int(means_v.shape[0])}, "
                f"max_scale={float(scales_v.detach().max().item()) if scales_v.numel() else 0.0:.6g}, "
                f"min_depth={float(depth[valid].detach().min().item()) if valid.any() else 0.0:.6g}, "
                f"max_screen_radius={float(max_screen_radius):.3g})."
            )
            return render_single_view(
                means,
                sh0,
                scales,
                opacity,
                intrinsic,
                extrinsic,
                height,
                width,
                exists=exists,
                mask_points=mask_points,
                patch_radius=patch_radius,
                mask_radius=mask_radius,
            )
    if mask_points is None:
        mask_points = means
    mask = _projection_mask_from_points(
        mask_points.float(),
        intrinsic,
        extrinsic,
        height,
        width,
        exists,
        radius=mask_radius,
    )
    return image.clamp(0.0, 1.0), mask


def render_single_view(
    means: torch.Tensor,
    sh0: torch.Tensor,
    scales: torch.Tensor,
    opacity: torch.Tensor,
    intrinsic: torch.Tensor,
    extrinsic: torch.Tensor,
    height: int,
    width: int,
    exists: torch.Tensor | None = None,
    mask_points: torch.Tensor | None = None,
    patch_radius: int = 2,
    mask_radius: int = 2,
    min_alpha: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = means.device
    dtype = torch.float32
    if exists is None:
        exists = torch.ones(means.shape[0], dtype=torch.bool, device=device)

    pixels, depth = _project_points(means, intrinsic, extrinsic)
    valid = (
        exists.bool()
        & torch.isfinite(pixels).all(dim=-1)
        & (depth > 0)
        & (pixels[:, 0] >= -patch_radius)
        & (pixels[:, 0] < width + patch_radius)
        & (pixels[:, 1] >= -patch_radius)
        & (pixels[:, 1] < height + patch_radius)
    )
    if not valid.any():
        image = torch.zeros((3, height, width), device=device, dtype=dtype)
        mask = torch.zeros((1, height, width), device=device, dtype=torch.bool)
        return image, mask

    pixels = pixels[valid].float()
    depth = depth[valid].float()
    colors = sh0_to_rgb(sh0[valid].float())
    alpha = opacity[valid].float().view(-1).clamp(0.0, 0.995)
    world_scale = scales[valid].float().mean(dim=-1).clamp_min(1e-5)
    focal = 0.5 * (intrinsic[0, 0].float().abs() + intrinsic[1, 1].float().abs())
    sigma = (focal * world_scale / depth.clamp_min(1e-4)).clamp(0.35, float(max(patch_radius, 1)))

    offsets = torch.arange(-patch_radius, patch_radius + 1, device=device, dtype=torch.float32)
    oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
    offsets_xy = torch.stack([ox.reshape(-1), oy.reshape(-1)], dim=-1)
    center_rounded = torch.round(pixels)
    coords = center_rounded[:, None, :] + offsets_xy[None, :, :]
    x = coords[..., 0].long()
    y = coords[..., 1].long()
    in_bounds = (x >= 0) & (x < width) & (y >= 0) & (y < height)

    delta = coords - pixels[:, None, :]
    dist2 = delta.pow(2).sum(dim=-1)
    weights = torch.exp(-0.5 * dist2 / sigma[:, None].pow(2)) * alpha[:, None]
    weights = torch.where(in_bounds, weights, torch.zeros_like(weights))

    flat_idx = (y.clamp(0, height - 1) * width + x.clamp(0, width - 1)).reshape(-1)
    flat_w = weights.reshape(-1)
    flat_c = (weights[..., None] * colors[:, None, :]).reshape(-1, 3)

    accum_w = torch.zeros((height * width,), device=device, dtype=dtype)
    accum_c = torch.zeros((height * width, 3), device=device, dtype=dtype)
    accum_w.scatter_add_(0, flat_idx, flat_w)
    accum_c.scatter_add_(0, flat_idx[:, None].expand(-1, 3), flat_c)

    alpha_img = accum_w.clamp(0.0, 1.0).view(height, width)
    color_img = (accum_c / accum_w.clamp_min(1e-6).unsqueeze(-1)).view(height, width, 3)
    image = (color_img * alpha_img.unsqueeze(-1)).permute(2, 0, 1).contiguous()
    if mask_points is None:
        mask_points = means
    mask = _projection_mask_from_points(
        mask_points.float(),
        intrinsic,
        extrinsic,
        height,
        width,
        exists,
        radius=mask_radius,
    )
    return image, mask


def render_batch_views(
    gaussians: dict[str, torch.Tensor],
    data_dict: dict,
    *,
    patch_radius: int,
    mask_size: int = 5,
    min_alpha: float = 1e-4,
    limit_views: int | None = None,
    backend: str = "diff_gaussian",
    znear: float = 0.01,
    zfar: float = 100.0,
    min_render_depth: float = 0.05,
    max_screen_radius: float = 64.0,
) -> list[dict[str, torch.Tensor | str | int]]:
    prefixes = camera_prefixes(data_dict)
    if limit_views is not None and limit_views > 0:
        prefixes = prefixes[:limit_views]
    if not prefixes:
        return []

    means = gaussians["means"]
    mask_points = gaussians.get("mask_points", means)
    sh0 = gaussians["sh0"]
    quats = gaussians["q"]
    scales = gaussians["s"]
    opacity = gaussians["o"]
    scene_exists = data_dict["scene_exists"][:, 0].bool()

    rendered = []
    mask_radius = int(mask_size) // 2
    B = means.shape[0]
    for b in range(B):
        for prefix in prefixes:
            exists_key = f"{prefix}_exists"
            if exists_key in data_dict and not bool(data_dict[exists_key][b].item()):
                continue
            target = _as_float_rgb(data_dict[f"{prefix}_initial_rgb"][b]).permute(2, 0, 1).contiguous()
            H, W = target.shape[-2:]
            if backend == "auto":
                backend_to_use = "diff_gaussian" if means.is_cuda else "torch"
            else:
                backend_to_use = backend
            if backend_to_use == "diff_gaussian":
                image, mask = render_single_view_diff_gaussian(
                    means[b],
                    sh0[b],
                    quats[b],
                    scales[b],
                    opacity[b],
                    data_dict[f"{prefix}_intrinsic"][b],
                    data_dict[f"{prefix}_extrinsic"][b],
                    H,
                    W,
                    exists=scene_exists[b],
                    mask_points=mask_points[b],
                    patch_radius=patch_radius,
                    mask_radius=mask_radius,
                    znear=znear,
                    zfar=zfar,
                    min_render_depth=min_render_depth,
                    max_screen_radius=max_screen_radius,
                )
            elif backend_to_use == "torch":
                image, mask = render_single_view(
                    means[b],
                    sh0[b],
                    scales[b],
                    opacity[b],
                    data_dict[f"{prefix}_intrinsic"][b],
                    data_dict[f"{prefix}_extrinsic"][b],
                    H,
                    W,
                    exists=scene_exists[b],
                    mask_points=mask_points[b],
                    patch_radius=patch_radius,
                    mask_radius=mask_radius,
                    min_alpha=min_alpha,
                )
            else:
                raise ValueError(f"Unknown gaussian renderer backend: {backend}")
            rendered.append(
                {
                    "image": image,
                    "target": target,
                    "mask": mask,
                    "prefix": prefix,
                    "batch_index": b,
                    "backend": backend_to_use,
                }
            )
    return rendered


def gaussian_image_loss(
    gaussians: dict[str, torch.Tensor],
    data_dict: dict,
    *,
    patch_radius: int,
    mask_size: int,
    ssim_weight: float,
    use_projection_mask: bool,
    limit_views: int | None = None,
    backend: str = "diff_gaussian",
    znear: float = 0.01,
    zfar: float = 100.0,
    min_render_depth: float = 0.05,
    max_screen_radius: float = 64.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    views = render_batch_views(
        gaussians,
        data_dict,
        patch_radius=patch_radius,
        mask_size=mask_size,
        limit_views=limit_views,
        backend=backend,
        znear=znear,
        zfar=zfar,
        min_render_depth=min_render_depth,
        max_screen_radius=max_screen_radius,
    )
    if not views:
        zero = gaussians["means"].new_zeros(())
        return zero, {
            "gaussian/l1": 0.0,
            "gaussian/dssim": 0.0,
            "gaussian/mask_fraction": 0.0,
            "gaussian/scale_mean": 0.0,
            "gaussian/scale_max": 0.0,
            "gaussian/num_views": 0.0,
            "gaussian/backend_diff_gaussian": 0.0,
        }

    losses = []
    l1_vals = []
    dssim_vals = []
    mask_fracs = []
    diff_backend_count = 0
    for view in views:
        image = view["image"]
        target = view["target"]
        projection_mask = view["mask"]
        if use_projection_mask:
            loss_mask = projection_mask
        else:
            loss_mask = torch.ones(
                (1, image.shape[-2], image.shape[-1]),
                device=image.device,
                dtype=torch.bool,
            )
        if loss_mask.any():
            l1 = (image - target).abs()[loss_mask.expand_as(image)].mean()
            dssim = dssim_loss(image, target, loss_mask)
            loss = (1.0 - ssim_weight) * l1 + ssim_weight * dssim
        else:
            l1 = image.new_zeros(())
            dssim = image.new_zeros(())
            loss = image.new_zeros(())
        losses.append(loss)
        l1_vals.append(float(l1.detach().item()))
        dssim_vals.append(float(dssim.detach().item()))
        mask_fracs.append(float(loss_mask.float().mean().detach().item()))
        if view.get("backend") == "diff_gaussian":
            diff_backend_count += 1

    total = torch.stack(losses).mean()
    return total, {
        "gaussian/l1": sum(l1_vals) / len(l1_vals),
        "gaussian/dssim": sum(dssim_vals) / len(dssim_vals),
        "gaussian/mask_fraction": sum(mask_fracs) / len(mask_fracs),
        "gaussian/scale_mean": float(gaussians["s"].detach().float().mean().item()),
        "gaussian/scale_max": float(gaussians["s"].detach().float().max().item()),
        "gaussian/use_projection_mask": float(bool(use_projection_mask)),
        "gaussian/num_views": float(len(views)),
        "gaussian/backend_diff_gaussian": float(diff_backend_count) / float(len(views)),
    }


@torch.no_grad()
def save_gaussian_renders(
    gaussians: dict[str, torch.Tensor],
    data_dict: dict,
    output_dir: str | os.PathLike[str],
    *,
    tag: str,
    step: int,
    patch_radius: int,
    mask_size: int,
    max_samples: int,
    save_ply: bool = True,
    backend: str = "diff_gaussian",
    znear: float = 0.01,
    zfar: float = 100.0,
    min_render_depth: float = 0.05,
    max_screen_radius: float = 64.0,
) -> int:
    if max_samples <= 0:
        return 0
    views = render_batch_views(
        gaussians,
        data_dict,
        patch_radius=patch_radius,
        mask_size=mask_size,
        limit_views=None,
        backend=backend,
        znear=znear,
        zfar=zfar,
        min_render_depth=min_render_depth,
        max_screen_radius=max_screen_radius,
    )
    if not views:
        return 0
    out_root = Path(output_dir) / "gaussian_renders" / tag / f"step_{int(step):08d}"
    out_root.mkdir(parents=True, exist_ok=True)

    keys = data_dict.get("__key__", None)
    scene_exists = data_dict["scene_exists"][:, 0].bool()
    saved = 0
    saved_ply_batches: set[int] = set()
    for view in views:
        b = int(view["batch_index"])
        sample_key = str(keys[b]) if isinstance(keys, (list, tuple)) and b < len(keys) else f"sample{b}"
        safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_key)[-120:]
        if save_ply and b not in saved_ply_batches:
            sample_gaussians = {
                name: value[b]
                for name, value in gaussians.items()
                if isinstance(value, torch.Tensor)
            }
            save_gaussian_ply(
                sample_gaussians,
                scene_exists[b],
                out_root / f"{safe_key}_gaussians.ply",
            )
            saved_ply_batches.add(b)
        prefix = str(view["prefix"])
        save_image(view["image"].clamp(0.0, 1.0), out_root / f"{safe_key}_{prefix}_pred.png")
        save_image(view["target"].clamp(0.0, 1.0), out_root / f"{safe_key}_{prefix}_target.png")
        save_image(view["mask"].float(), out_root / f"{safe_key}_{prefix}_mask.png")
        saved += 1
        if saved >= max_samples:
            break
    return saved
