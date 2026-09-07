#!/usr/bin/env bash
set -eo pipefail

# Environment-only compatibility build for GINO on RTX 5070 / CUDA 13.
# It does not alter GINO radii, graph construction, architecture, loss,
# optimizer, or normalization. Third-party archives must be pre-staged and
# independently SHA-verified; this script performs no network download.

source /home/xyh/miniconda3/etc/profile.d/conda.sh
conda activate g2-gino-open3d-src

src=/home/xyh/myCodeGitOnly/external/g2/open3d-src-1e7b174
build=/home/xyh/myCodeGitOnly/external/g2/open3d-build-1e7b174-torch29-cu130-sm120-cudart-dev
cuda_root=/home/xyh/miniconda3/envs/g2-gino-open3d-src/lib/python3.12/site-packages/nvidia/cu13
cccl_root=/home/xyh/myCodeGitOnly/external/g2/cccl-v2.8.5
expected_open3d_commit=1e7b17438687a0b0c1e5a7187321ac7044afe275
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
npp_package_version=$(python -c 'import importlib.metadata; print(importlib.metadata.version("nvidia-npp"))' 2>/dev/null || true)
[[ "$npp_package_version" == "13.0.3.3" ]] || {
  echo "FAIL_NPP_VERSION expected=13.0.3.3 actual=${npp_package_version:-MISSING}"
  exit 2
}

[[ "$(git -C "$src" rev-parse HEAD)" == "$expected_open3d_commit" ]] || {
  echo FAIL_OPEN3D_COMMIT
  exit 2
}
[[ "$(sha256sum /home/xyh/myCodeGitOnly/external/g2/cccl-v2.8.5.tar.gz | cut -d' ' -f1)" == "226d4794c5f0cbb6040d022a9bcf8be40cbc2fa147b633940d9b439529ce6a4b" ]] || {
  echo FAIL_CCCL_ARCHIVE_SHA
  exit 2
}

declare -A archives=(
  ["$src/3rdparty_downloads/zlib/v1.2.13.tar.gz"]="1525952a0a567581792613a9723333d7f8cc20b87a81f920fb8bc7e3f2251428"
  ["$src/3rdparty_downloads/uvatlas/DirectX-Headers-v1.606.3.tar.gz"]="bf0183981e505336e918609374907c934b99eb61c0826d75a5649f41568abc4b"
  ["$src/3rdparty_downloads/uvatlas/DirectXMath-may2022.tar.gz"]="b2c5b419ca2c567860f7c204c9c0890573e8a58c8d877473e4f3ba6b851ca4ce"
  ["$src/3rdparty_downloads/uvatlas/may2022.tar.gz"]="591516913a0f3c381f1fd01647cb1b8d1eeade575d1c726ae8f5dd9f83b81754"
  ["$src/3rdparty_downloads/eigen/eigen-da7909592376c893dabbc4b6453a8ffe46b1eb8e.tar.gz"]="2b2afe0ad9f2148e41dedf4f562901606ccd2c47b98422a4e707d6bf8d1a4416"
)
for artifact in "${!archives[@]}"; do
  [[ -f "$artifact" ]] || { echo "FAIL_MISSING_ARCHIVE $artifact"; exit 2; }
  [[ "$(sha256sum "$artifact" | cut -d' ' -f1)" == "${archives[$artifact]}" ]] || {
    echo "FAIL_ARCHIVE_SHA $artifact"
    exit 2
  }
done

cp "$repo_root/patches/stdgpu_2588168_cuda13_clock_rate.patch" \
  "$src/3rdparty/stdgpu/stdgpu_2588168_cuda13_clock_rate.patch"
for patch_relative in \
  patches/open3d_v019_cuda_sm120_arch_translation.patch \
  patches/open3d_v019_cuda13_shared_cublas.patch \
  patches/open3d_v019_uvatlas_archive_downloads.patch \
  patches/open3d_v019_eigen_gitlab_api_archive.patch \
  patches/open3d_v019_glfw_disable_wayland.patch \
  patches/open3d_v019_cuda13_cccl_stdgpu.patch \
  patches/open3d_v019_stdgpu_cuda13_patch_hook.patch; do
  patch_path="$repo_root/$patch_relative"
  if git -C "$src" apply --check "$patch_path"; then
    git -C "$src" apply "$patch_path"
  elif git -C "$src" apply --reverse --check "$patch_path"; then
    :
  else
    echo "FAIL_PATCH_STATE $patch_relative"
    exit 2
  fi
done

export CUDA_HOME="$cuda_root"
export PATH="$cuda_root/bin:$PATH"
export LD_LIBRARY_PATH="$cuda_root/lib:${LD_LIBRARY_PATH:-}"
[[ -f "$cuda_root/include/npp.h" ]] || { echo "FAIL_MISSING_NPP_HEADER $cuda_root/include/npp.h"; exit 2; }
export CPATH="$cuda_root/include:$cccl_root/thrust:$cccl_root/cub:$cccl_root/libcudacxx/include:$CONDA_PREFIX/include:${CPATH:-}"

[[ -e "$cuda_root/lib64" ]] || ln -s lib "$cuda_root/lib64"
for pair in \
  libcudart.so:libcudart.so.13 \
  libcublas.so:libcublas.so.13 \
  libcublasLt.so:libcublasLt.so.13 \
  libcufft.so:libcufft.so.12 \
  libcufftw.so:libcufftw.so.12 \
  libcurand.so:libcurand.so.10 \
  libcusolver.so:libcusolver.so.12 \
  libcusolverMg.so:libcusolverMg.so.12 \
  libcusparse.so:libcusparse.so.12 \
  libnvrtc.so:libnvrtc.so.13; do
  link_name="${pair%%:*}"
  target_name="${pair#*:}"
  [[ -e "$cuda_root/lib/$link_name" ]] || ln -s "$target_name" "$cuda_root/lib/$link_name"
done
[[ -e "$cuda_root/lib/libcuda.so" ]] || ln -s /usr/lib/wsl/lib/libcuda.so "$cuda_root/lib/libcuda.so"
for npp_lib in "$cuda_root"/lib/libnpp*.so.13; do
  [[ -e "$npp_lib" ]] || continue
  npp_link="${npp_lib%.13}"
  [[ -e "$npp_link" ]] || ln -s "$(basename "$npp_lib")" "$npp_link"
done

nice -n 10 cmake -S "$src" -B "$build" -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DDEVELOPER_BUILD=OFF \
  -DBUILD_CUDA_MODULE=ON \
  -DBUILD_WITH_CUDA_STATIC=OFF \
  -DCMAKE_CUDA_COMPILER="$cuda_root/bin/nvcc" \
  -DCUDAToolkit_ROOT="$cuda_root" \
  -DG2_CCCL_ROOT="$cccl_root" \
  -DCUDA_NVRTC_LIB="$cuda_root/lib/libnvrtc.so" \
  -DCMAKE_CUDA_FLAGS="-I$cccl_root/thrust -I$cccl_root/cub -I$cccl_root/libcudacxx/include" \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared \
  -DBUILD_PYTHON_MODULE=ON \
  -DBUILD_PYTORCH_OPS=ON \
  -DBUILD_TENSORFLOW_OPS=OFF \
  -DBUNDLE_OPEN3D_ML=OFF \
  -DGLIBCXX_USE_CXX11_ABI=ON \
  -DPython3_EXECUTABLE="$(which python)" \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  -DBUILD_GUI=OFF \
  -DBUILD_WEBRTC=OFF \
  -DBUILD_JUPYTER_EXTENSION=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_UNIT_TESTS=OFF \
  -DBUILD_BENCHMARKS=OFF \
  -DBUILD_ISPC_MODULE=OFF \
  -DBUILD_AZURE_KINECT=OFF \
  -DBUILD_LIBREALSENSE=OFF

nice -n 10 cmake --build "$build" --target pip-package --parallel 2
find "$build" -type f -name '*.whl' -print -exec sha256sum {} \;
