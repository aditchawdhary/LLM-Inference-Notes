# Installing VLLM on Rocm hardware

- MI 300x1-192gb
- Running https://huggingface.co/microsoft/Phi-4-mini-flash-reasoning/tree/main
- https://docs.vllm.ai/en/v0.6.5/getting_started/amd-installation.html

1. For installing PyTorch, you can start from a fresh docker image,
```
sudo docker run -it \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  --shm-size 80G \
  rocm/pytorch:rocm6.2_ubuntu20.04_py3.9_pytorch_release_2.3.0
```

2. Install Triton flash attention for ROCm
```
python3 -m pip install ninja cmake wheel pybind11
pip uninstall -y triton
git clone https://github.com/OpenAI/triton.git
cd triton
git checkout e192dba
cd python
pip3 install .
cd ../..
```

3. Optionally, if you choose to use CK flash attention, you can install flash attention for ROCm
```
git clone https://github.com/ROCm/flash-attention.git
cd flash-attention
git checkout 3cea2fb
git submodule update --init
GPU_ARCHS="gfx90a" python3 setup.py install
cd ..
```

4. Build vLLM.
```
pip install --upgrade pip

# Install PyTorch
pip uninstall torch -y
pip install --no-cache-dir --pre torch==2.6.0.dev20240918 --index-url https://download.pytorch.org/whl/nightly/rocm6.2

# Build & install AMD SMI
pip install /opt/rocm/share/amd_smi

# Install dependencies
pip install --upgrade numba scipy huggingface-hub[cli]
pip install "numpy<2"
pip install -r requirements-rocm.txt

# Build vLLM for MI210/MI250/MI300.
export PYTORCH_ROCM_ARCH="gfx90a;gfx942"
python3 setup.py develop
```
