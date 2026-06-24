#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-pointwm}"
ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"

git submodule update --init --recursive third_party/diff-gaussian-rasterization

if command -v nvcc >/dev/null 2>&1; then
  echo "Found nvcc at $(command -v nvcc); skipping NVIDIA conda package install."
  nvcc --version
else
  conda install -n "${CONDA_ENV}" -c nvidia \
    cuda-nvcc=12.4 \
    cuda-cccl=12.4.127 \
    cuda-cudart-dev=12.4.127 \
    -y
fi

TORCH_CUDA_ARCH_LIST="${ARCH_LIST}" MAX_JOBS="${MAX_JOBS:-8}" \
  conda run -n "${CONDA_ENV}" \
  pip install --no-build-isolation -e third_party/diff-gaussian-rasterization

SMOKE_PY="$(mktemp)"
cat > "${SMOKE_PY}" <<'PY'
import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

device = "cuda"
H = W = 64
fx = fy = 80.0
tanfovx = W / (2.0 * fx)
tanfovy = H / (2.0 * fy)
znear = 0.01
zfar = 100.0

proj = torch.zeros(4, 4, device=device)
proj[0, 0] = 1.0 / tanfovx
proj[1, 1] = 1.0 / tanfovy
proj[3, 2] = 1.0
proj[2, 2] = zfar / (zfar - znear)
proj[2, 3] = -(zfar * znear) / (zfar - znear)
view = torch.eye(4, device=device)
full = view @ proj.t().contiguous()

means = torch.tensor([[0.0, 0.0, 2.0], [0.1, 0.0, 2.0]], device=device, requires_grad=True)
means2d = torch.zeros_like(means, requires_grad=True)
colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device, requires_grad=True)
opacities = torch.full((2, 1), 0.8, device=device, requires_grad=True)
scales = torch.full((2, 3), 0.05, device=device, requires_grad=True)
rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], device=device)

settings = GaussianRasterizationSettings(
    image_height=H,
    image_width=W,
    tanfovx=tanfovx,
    tanfovy=tanfovy,
    bg=torch.zeros(3, device=device),
    scale_modifier=1.0,
    viewmatrix=view,
    projmatrix=full,
    sh_degree=0,
    campos=torch.zeros(3, device=device),
    prefiltered=False,
    debug=False,
)
rasterizer = GaussianRasterizer(settings)
image, radii = rasterizer(
    means3D=means,
    means2D=means2d,
    colors_precomp=colors,
    opacities=opacities,
    scales=scales,
    rotations=rotations,
)
loss = image.sum()
loss.backward()
print("diff_gaussian smoke ok", tuple(image.shape), radii.tolist(), float(means.grad.abs().sum()))
PY
conda run -n "${CONDA_ENV}" python "${SMOKE_PY}"
rm -f "${SMOKE_PY}"
