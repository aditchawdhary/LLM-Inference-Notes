# LLM-Inference-Notes
LLM inference notes

## TensorRT-LLM

## Machines I have installed this on in the Cloud:
1. 2x H100 NVL
2. 1x RTX PRO 6000 WS
3. 1x H100 NVL
4. 1x H200 NVL - India


```bash


# Commands for installation:
pip3 install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

sudo apt-get -y install libopenmpi-dev

# Activated conda/uv virtual environment at /venv/main
(main) root@C.25238311:/workspace$ python --version
# Python 3.12.11
(main) root@C.25238311:/workspace$ pip3 --version
# pip 25.1.1 from /venv/main/lib/python3.12/site-packages/pip (python 3.12)
(main) root@C.25238311:/workspace$ pip3 install tensorrt-llm --extra-index-url https://pypi.nvidia.com/
: '
Looking in indexes: https://pypi.org/simple, https://pypi.nvidia.com/
Collecting tensorrt-llm
Downloading https://pypi.nvidia.com/tensorrt-llm/tensorrt_llm-0.21.0-cp312-cp312-linux_x86_64.whl (3932.9 MB)
Looking in indexes: https://pypi.org/simple, https://pypi.nvidia.com/
Collecting tensorrt-llm
  Downloading https://pypi.nvidia.com/tensorrt-llm/tensorrt_llm-0.21.0-cp312-cp312-linux_x86_64.whl (3932.9 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.9/3.9 GB 239.0 MB/s eta 0:00:00
Collecting accelerate>=0.25.0 (from tensorrt-llm)
  Downloading accelerate-1.10.0-py3-none-any.whl.metadata (19 kB)
Collecting build (from tensorrt-llm)
  Downloading build-1.3.0-py3-none-any.whl.metadata (5.6 kB)
Collecting colored (from tensorrt-llm)
  Downloading colored-2.3.1-py3-none-any.whl.metadata (3.6 kB)
Collecting cuda-python (from tensorrt-llm)
  Downloading cuda_python-13.0.1-py3-none-any.whl.metadata (4.7 kB)
Collecting diffusers>=0.27.0 (from tensorrt-llm)
  Downloading diffusers-0.35.1-py3-none-any.whl.metadata (20 kB)
Collecting lark (from tensorrt-llm)
  Downloading lark-1.2.2-py3-none-any.whl.metadata (1.8 kB)
Collecting mpi4py (from tensorrt-llm)
  Downloading mpi4py-4.1.0-cp312-cp312-manylinux1_x86_64.manylinux_2_5_x86_64.whl.metadata (16 kB)
Collecting numpy<2 (from tensorrt-llm)
  Downloading numpy-1.26.4-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (61 kB)
Collecting onnx>=1.12.0 (from tensorrt-llm)
  Downloading onnx-1.18.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.9 kB)
Collecting onnx_graphsurgeon>=0.5.2 (from tensorrt-llm)
  Downloading https://pypi.nvidia.com/onnx-graphsurgeon/onnx_graphsurgeon-0.5.8-py2.py3-none-any.whl (57 kB)
Collecting openai (from tensorrt-llm)
  Downloading openai-1.101.0-py3-none-any.whl.metadata (29 kB)
Collecting polygraphy (from tensorrt-llm)
  Downloading https://pypi.nvidia.com/polygraphy/polygraphy-0.49.26-py2.py3-none-any.whl (372 kB)
Requirement already satisfied: psutil in /venv/main/lib/python3.12/site-packages (from tensorrt-llm) (7.0.0)
Collecting nvidia-ml-py>=12 (from tensorrt-llm)
  Downloading nvidia_ml_py-13.580.65-py3-none-any.whl.metadata (9.6 kB)
Collecting pynvml>=12.0.0 (from tensorrt-llm)
  Downloading pynvml-12.0.0-py3-none-any.whl.metadata (5.4 kB)
Collecting pulp (from tensorrt-llm)
  Downloading pulp-3.2.2-py3-none-any.whl.metadata (6.9 kB)
Collecting pandas (from tensorrt-llm)
  Downloading pandas-2.3.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (91 kB)
Collecting h5py==3.12.1 (from tensorrt-llm)
  Downloading h5py-3.12.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (2.5 kB)
Collecting StrEnum (from tensorrt-llm)
  Downloading StrEnum-0.4.15-py3-none-any.whl.metadata (5.3 kB)
Collecting sentencepiece>=0.1.99 (from tensorrt-llm)
  Downloading sentencepiece-0.2.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (10 kB)
Collecting tensorrt~=10.11.0 (from tensorrt-llm)
  Downloading https://pypi.nvidia.com/tensorrt/tensorrt-10.11.0.33.tar.gz (40 kB)
  Preparing metadata (setup.py) ... done
Collecting torch<=2.8.0a0,>=2.7.1 (from tensorrt-llm)
  Downloading torch-2.7.1-cp312-cp312-manylinux_2_28_x86_64.whl.metadata (29 kB)
Collecting torchvision (from tensorrt-llm)
  Downloading torchvision-0.23.0-cp312-cp312-manylinux_2_28_x86_64.whl.metadata (6.1 kB)
Collecting nvidia-modelopt~=0.31.0 (from nvidia-modelopt[torch]~=0.31.0->tensorrt-llm)
  Downloading https://pypi.nvidia.com/nvidia-modelopt/nvidia_modelopt-0.31.0-py3-none-manylinux_2_28_x86_64.whl (717 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 717.2/717.2 kB 313.1 MB/s eta 0:00:00
Collecting nvidia-nccl-cu12 (from tensorrt-llm)
  Downloading https://pypi.nvidia.com/nvidia-nccl-cu12/nvidia_nccl_cu12-2.27.7-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (322.5 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 322.5/322.5 MB 227.6 MB/s eta 0:00:00
Collecting nvidia-cuda-nvrtc-cu12 (from tensorrt-llm)
  Downloading https://pypi.nvidia.com/nvidia-cuda-nvrtc-cu12/nvidia_cuda_nvrtc_cu12-12.9.86-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl (89.6 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 89.6/89.6 MB 254.4 MB/s eta 0:00:00
Collecting transformers~=4.51.1 (from tensorrt-llm)
  Downloading transformers-4.51.3-py3-none-any.whl.metadata (38 kB)
Collecting pydantic>=2.9.1 (from tensorrt-llm)
  Downloading pydantic-2.11.7-py3-none-any.whl.metadata (67 kB)
Collecting pillow==10.3.0 (from tensorrt-llm)
  Downloading pillow-10.3.0-cp312-cp312-manylinux_2_28_x86_64.whl.metadata (9.2 kB)
Requirement already satisfied: wheel<=0.45.1 in /venv/main/lib/python3.12/site-packages (from tensorrt-llm) (0.45.1)
Collecting optimum (from tensorrt-llm)
  Downloading optimum-1.27.0-py3-none-any.whl.metadata (16 kB)
Collecting datasets==3.1.0 (from tensorrt-llm)
  Downloading datasets-3.1.0-py3-none-any.whl.metadata (20 kB)
Collecting evaluate (from tensorrt-llm)
  Downloading evaluate-0.4.5-py3-none-any.whl.metadata (9.5 kB)
Collecting mpmath>=1.3.0 (from tensorrt-llm)
  Downloading mpmath-1.3.0-py3-none-any.whl.metadata (8.6 kB)
Collecting click (from tensorrt-llm)
  Downloading click-8.2.1-py3-none-any.whl.metadata (2.5 kB)
Collecting click_option_group (from tensorrt-llm)
  Downloading click_option_group-0.5.7-py3-none-any.whl.metadata (5.8 kB)
Collecting aenum (from tensorrt-llm)
  Downloading aenum-3.1.16-py3-none-any.whl.metadata (3.8 kB)
Requirement already satisfied: pyzmq in /venv/main/lib/python3.12/site-packages (from tensorrt-llm) (27.0.0)
Collecting fastapi==0.115.4 (from tensorrt-llm)
  Downloading fastapi-0.115.4-py3-none-any.whl.metadata (27 kB)
Collecting uvicorn (from tensorrt-llm)
  Downloading uvicorn-0.35.0-py3-none-any.whl.metadata (6.5 kB)
Collecting setuptools<80 (from tensorrt-llm)
  Downloading setuptools-79.0.1-py3-none-any.whl.metadata (6.5 kB)
Collecting ordered-set (from tensorrt-llm)
  Downloading ordered_set-4.1.0-py3-none-any.whl.metadata (5.3 kB)
Collecting peft (from tensorrt-llm)
  Downloading peft-0.17.1-py3-none-any.whl.metadata (14 kB)
Collecting einops (from tensorrt-llm)
  Downloading einops-0.8.1-py3-none-any.whl.metadata (13 kB)
Collecting flashinfer-python==0.2.5 (from tensorrt-llm)
  Downloading flashinfer_python-0.2.5.tar.gz (2.5 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.5/2.5 MB 121.8 MB/s eta 0:00:00
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
Collecting opencv-python-headless (from tensorrt-llm)
  Downloading opencv_python_headless-4.12.0.88-cp37-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (19 kB)
Collecting xgrammar==0.1.18 (from tensorrt-llm)
  Downloading xgrammar-0.1.18-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.6 kB)
Collecting backoff (from tensorrt-llm)
  Downloading backoff-2.2.1-py3-none-any.whl.metadata (14 kB)
Collecting nvtx (from tensorrt-llm)
  Downloading https://pypi.nvidia.com/nvtx/nvtx-0.2.13-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (545 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 545.6/545.6 kB 248.6 MB/s eta 0:00:00
Collecting matplotlib (from tensorrt-llm)
  Downloading matplotlib-3.10.5-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (11 kB)
Collecting meson (from tensorrt-llm)
  Downloading meson-1.8.4-py3-none-any.whl.metadata (1.8 kB)
Collecting ninja (from tensorrt-llm)
  Downloading ninja-1.13.0-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (5.1 kB)
Collecting etcd3 (from tensorrt-llm)
  Downloading etcd3-0.12.0.tar.gz (63 kB)
  Preparing metadata (setup.py) ... done
Collecting blake3 (from tensorrt-llm)
  Downloading blake3-1.0.5-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.2 kB)
Collecting llguidance==0.7.29 (from tensorrt-llm)
  Downloading llguidance-0.7.29-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (9.9 kB)
Collecting triton==3.3.1 (from tensorrt-llm)
  Downloading triton-3.3.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (1.5 kB)
Collecting h11>=0.16.0 (from tensorrt-llm)
  Downloading h11-0.16.0-py3-none-any.whl.metadata (8.3 kB)
Requirement already satisfied: tornado>=6.5.0 in /venv/main/lib/python3.12/site-packages (from tensorrt-llm) (6.5.1)
Requirement already satisfied: filelock in /venv/main/lib/python3.12/site-packages (from datasets==3.1.0->tensorrt-llm) (3.18.0)
Collecting pyarrow>=15.0.0 (from datasets==3.1.0->tensorrt-llm)
  Downloading pyarrow-21.0.0-cp312-cp312-manylinux_2_28_x86_64.whl.metadata (3.3 kB)
Collecting dill<0.3.9,>=0.3.0 (from datasets==3.1.0->tensorrt-llm)
  Downloading dill-0.3.8-py3-none-any.whl.metadata (10 kB)
Requirement already satisfied: requests>=2.32.2 in /venv/main/lib/python3.12/site-packages (from datasets==3.1.0->tensorrt-llm) (2.32.4)
Requirement already satisfied: tqdm>=4.66.3 in /venv/main/lib/python3.12/site-packages (from datasets==3.1.0->tensorrt-llm) (4.67.1)
Collecting xxhash (from datasets==3.1.0->tensorrt-llm)
  Downloading xxhash-3.5.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (12 kB)
Collecting multiprocess<0.70.17 (from datasets==3.1.0->tensorrt-llm)
  Downloading multiprocess-0.70.16-py312-none-any.whl.metadata (7.2 kB)
Collecting fsspec<=2024.9.0,>=2023.1.0 (from fsspec[http]<=2024.9.0,>=2023.1.0->datasets==3.1.0->tensorrt-llm)
  Downloading fsspec-2024.9.0-py3-none-any.whl.metadata (11 kB)
Collecting aiohttp (from datasets==3.1.0->tensorrt-llm)
  Downloading aiohttp-3.12.15-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (7.7 kB)
Requirement already satisfied: huggingface-hub>=0.23.0 in /venv/main/lib/python3.12/site-packages (from datasets==3.1.0->tensorrt-llm) (0.33.1)
Requirement already satisfied: packaging in /venv/main/lib/python3.12/site-packages (from datasets==3.1.0->tensorrt-llm) (25.0)
Requirement already satisfied: pyyaml>=5.1 in /venv/main/lib/python3.12/site-packages (from datasets==3.1.0->tensorrt-llm) (6.0.2)
Collecting starlette<0.42.0,>=0.40.0 (from fastapi==0.115.4->tensorrt-llm)
  Downloading starlette-0.41.3-py3-none-any.whl.metadata (6.0 kB)
Requirement already satisfied: typing-extensions>=4.8.0 in /venv/main/lib/python3.12/site-packages (from fastapi==0.115.4->tensorrt-llm) (4.14.0)
Collecting tiktoken (from xgrammar==0.1.18->tensorrt-llm)
  Downloading tiktoken-0.11.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.7 kB)
Collecting nvidia-modelopt-core==0.31.0 (from nvidia-modelopt~=0.31.0->nvidia-modelopt[torch]~=0.31.0->tensorrt-llm)
  Downloading https://pypi.nvidia.com/nvidia-modelopt-core/nvidia_modelopt_core-0.31.0-cp312-cp312-manylinux_2_28_x86_64.whl (1.4 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.4/1.4 MB 395.5 MB/s eta 0:00:00
Collecting rich (from nvidia-modelopt~=0.31.0->nvidia-modelopt[torch]~=0.31.0->tensorrt-llm)
  Downloading rich-14.1.0-py3-none-any.whl.metadata (18 kB)
Collecting scipy (from nvidia-modelopt~=0.31.0->nvidia-modelopt[torch]~=0.31.0->tensorrt-llm)
  Downloading scipy-1.16.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (61 kB)
Collecting regex (from nvidia-modelopt[torch]~=0.31.0->tensorrt-llm)
  Downloading regex-2025.7.34-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (40 kB)
Collecting safetensors (from nvidia-modelopt[torch]~=0.31.0->tensorrt-llm)
  Downloading safetensors-0.6.2-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.1 kB)
Collecting torchprofile>=0.0.4 (from nvidia-modelopt[torch]~=0.31.0->tensorrt-llm)
  Downloading torchprofile-0.0.4-py3-none-any.whl.metadata (303 bytes)
Collecting annotated-types>=0.6.0 (from pydantic>=2.9.1->tensorrt-llm)
  Downloading annotated_types-0.7.0-py3-none-any.whl.metadata (15 kB)
Collecting pydantic-core==2.33.2 (from pydantic>=2.9.1->tensorrt-llm)
  Downloading pydantic_core-2.33.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.8 kB)
Collecting typing-inspection>=0.4.0 (from pydantic>=2.9.1->tensorrt-llm)
  Downloading typing_inspection-0.4.1-py3-none-any.whl.metadata (2.6 kB)
Collecting anyio<5,>=3.4.0 (from starlette<0.42.0,>=0.40.0->fastapi==0.115.4->tensorrt-llm)
  Downloading anyio-4.10.0-py3-none-any.whl.metadata (4.0 kB)
Requirement already satisfied: idna>=2.8 in /venv/main/lib/python3.12/site-packages (from anyio<5,>=3.4.0->starlette<0.42.0,>=0.40.0->fastapi==0.115.4->tensorrt-llm) (3.10)
Collecting sniffio>=1.1 (from anyio<5,>=3.4.0->starlette<0.42.0,>=0.40.0->fastapi==0.115.4->tensorrt-llm)
  Downloading sniffio-1.3.1-py3-none-any.whl.metadata (3.9 kB)
Collecting tensorrt_cu12==10.11.0.33 (from tensorrt~=10.11.0->tensorrt-llm)
  Downloading https://pypi.nvidia.com/tensorrt-cu12/tensorrt_cu12-10.11.0.33.tar.gz (18 kB)
  Preparing metadata (setup.py) ... done
Collecting tensorrt_cu12_libs==10.11.0.33 (from tensorrt_cu12==10.11.0.33->tensorrt~=10.11.0->tensorrt-llm)
  Downloading https://pypi.nvidia.com/tensorrt-cu12-libs/tensorrt_cu12_libs-10.11.0.33-py2.py3-none-manylinux_2_28_x86_64.whl (3095.4 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.1/3.1 GB 262.2 MB/s eta 0:00:00
Collecting tensorrt_cu12_bindings==10.11.0.33 (from tensorrt_cu12==10.11.0.33->tensorrt~=10.11.0->tensorrt-llm)
  Downloading https://pypi.nvidia.com/tensorrt-cu12-bindings/tensorrt_cu12_bindings-10.11.0.33-cp312-none-manylinux_2_28_x86_64.whl (1.2 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.2/1.2 MB 558.6 MB/s eta 0:00:00
Collecting nvidia-cuda-runtime-cu12 (from tensorrt_cu12_libs==10.11.0.33->tensorrt_cu12==10.11.0.33->tensorrt~=10.11.0->tensorrt-llm)
  Downloading https://pypi.nvidia.com/nvidia-cuda-runtime-cu12/nvidia_cuda_runtime_cu12-12.9.79-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (3.5 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.5/3.5 MB 404.7 MB/s eta 0:00:00
Collecting sympy>=1.13.3 (from torch<=2.8.0a0,>=2.7.1->tensorrt-llm)
  Downloading sympy-1.14.0-py3-none-any.whl.metadata (12 kB)
Collecting networkx (from torch<=2.8.0a0,>=2.7.1->tensorrt-llm)
  Downloading networkx-3.5-py3-none-any.whl.metadata (6.3 kB)
Collecting jinja2 (from torch<=2.8.0a0,>=2.7.1->tensorrt-llm)
  Downloading jinja2-3.1.6-py3-none-any.whl.metadata (2.9 kB)
Collecting nvidia-cuda-nvrtc-cu12 (from tensorrt-llm)
  Downloading https://pypi.nvidia.com/nvidia-cuda-nvrtc-cu12/nvidia_cuda_nvrtc_cu12-12.6.77-py3-none-manylinux2014_x86_64.whl (23.7 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 23.7/23.7 MB 197.2 MB/s eta 0:00:00
Collecting nvidia-cuda-runtime-cu12 (from tensorrt_cu12_libs==10.11.0.33->tensorrt_cu12==10.11.0.33->tensorrt~=10.11.0->tensorrt-llm)
  Downloading https://pypi.nvidia.com/nvidia-cuda-runtime-cu12/nvidia_cuda_runtime_cu12-12.6.77-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (897 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 897.7/897.7 kB 518.3 MB/s eta 0:00:00
Collecting nvidia-cuda-cupti-cu12==12.6.80 (from torch<=2.8.0a0,>=2.7.1->tensorrt-llm)
  Downloading nvidia_cuda_cupti_cu12-12.6.80-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.6 kB)
Collecting nvidia-cudnn-cu12==9.5.1.17 (from torch<=2.8.0a0,>=2.7.1->tensorrt-llm)
  Downloading https://pypi.nvidia.com/nvidia-cudnn-cu12/nvidia_cudnn_cu12-9.5.1.17-py3-none-manylinux_2_28_x86_64.whl (571.0 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 571.0/571.0 MB 257.5 MB/s eta 0:00:00
Collecting nvidia-cublas-cu12==12.6.4.1 (from torch<=2.8.0a0,>=2.7.1->tensorrt-llm)
  Downloading nvidia_cublas_cu12-12.6.4.1-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.5 kB)
Collecting nvidia-cufft-cu12==11.3.0.4 (from torch<=2.8.0a0,>=2.7.1->tensorrt-llm)
  Downloading https://pypi.nvidia.com/nvidia-cufft-cu12/nvidia_cufft_cu12-11.3.0.4-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (200.2 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 200.2/200.2 MB 260.2 MB/s eta 0:00:00
Collecting nvidia-curand-cu12==10.3.7.77 (from torch<=2.8.0a0,>=2.7.1->tensorrt-llm)
  Downloading https://pypi.nvidia.com/nvidia-curand-cu12/nvidia_curand_cu12-10.3.7.77-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (56.3 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 56.3/56.3 MB 263.5 MB/s eta 0:00:00
Collecting nvidia-cusolver-cu12==11.7.1.2 (from torch<=2.8.0a0,>=2.7.1->tensorrt-llm)
  Downloading https://pypi.nvidia.com/nvidia-cusolver-cu12/nvidia_cusolver_cu12-11.7.1.2-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (158.2 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 158.2/158.2 MB 266.3 MB/s eta 0:00:00
Collecting nvidia-cusparse-cu12==12.5.4.2 (from torch<=2.8.0a0,>=2.7.1->tensorrt-llm)
  Downloading https://pypi.nvidia.com/nvidia-cusparse-cu12/nvidia_cusparse_cu12-12.5.4.2-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (216.6 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 216.6/216.6 MB 265.7 MB/s eta 0:00:00
Collecting nvidia-cusparselt-cu12==0.6.3 (from torch<=2.8.0a0,>=2.7.1->tensorrt-llm)
  Downloading https://pypi.nvidia.com/nvidia-cusparselt-cu12/nvidia_cusparselt_cu12-0.6.3-py3-none-manylinux2014_x86_64.whl (156.8 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 156.8/156.8 MB 263.9 MB/s eta 0:00:00
Collecting nvidia-nccl-cu12 (from tensorrt-llm)
  Downloading https://pypi.nvidia.com/nvidia-nccl-cu12/nvidia_nccl_cu12-2.26.2-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (201.3 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 201.3/201.3 MB 261.9 MB/s eta 0:00:00
Collecting nvidia-nvtx-cu12==12.6.77 (from torch<=2.8.0a0,>=2.7.1->tensorrt-llm)
  Downloading https://pypi.nvidia.com/nvidia-nvtx-cu12/nvidia_nvtx_cu12-12.6.77-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (89 kB)
Collecting nvidia-nvjitlink-cu12==12.6.85 (from torch<=2.8.0a0,>=2.7.1->tensorrt-llm)
  Downloading https://pypi.nvidia.com/nvidia-nvjitlink-cu12/nvidia_nvjitlink_cu12-12.6.85-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl (19.7 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 19.7/19.7 MB 291.5 MB/s eta 0:00:00
Collecting nvidia-cufile-cu12==1.11.1.6 (from torch<=2.8.0a0,>=2.7.1->tensorrt-llm)
  Downloading https://pypi.nvidia.com/nvidia-cufile-cu12/nvidia_cufile_cu12-1.11.1.6-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (1.1 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.1/1.1 MB 395.2 MB/s eta 0:00:00
Collecting tokenizers<0.22,>=0.21 (from transformers~=4.51.1->tensorrt-llm)
  Downloading tokenizers-0.21.4-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.7 kB)
Requirement already satisfied: hf-xet<2.0.0,>=1.1.2 in /venv/main/lib/python3.12/site-packages (from huggingface-hub>=0.23.0->datasets==3.1.0->tensorrt-llm) (1.1.5)
Collecting aiohappyeyeballs>=2.5.0 (from aiohttp->datasets==3.1.0->tensorrt-llm)
  Downloading aiohappyeyeballs-2.6.1-py3-none-any.whl.metadata (5.9 kB)
Collecting aiosignal>=1.4.0 (from aiohttp->datasets==3.1.0->tensorrt-llm)
  Downloading aiosignal-1.4.0-py3-none-any.whl.metadata (3.7 kB)
Collecting attrs>=17.3.0 (from aiohttp->datasets==3.1.0->tensorrt-llm)
  Downloading attrs-25.3.0-py3-none-any.whl.metadata (10 kB)
Collecting frozenlist>=1.1.1 (from aiohttp->datasets==3.1.0->tensorrt-llm)
  Downloading frozenlist-1.7.0-cp312-cp312-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (18 kB)
Collecting multidict<7.0,>=4.5 (from aiohttp->datasets==3.1.0->tensorrt-llm)
  Downloading multidict-6.6.4-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (5.3 kB)
Collecting propcache>=0.2.0 (from aiohttp->datasets==3.1.0->tensorrt-llm)
  Downloading propcache-0.3.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (12 kB)
Collecting yarl<2.0,>=1.17.0 (from aiohttp->datasets==3.1.0->tensorrt-llm)
  Downloading yarl-1.20.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (73 kB)
Collecting importlib_metadata (from diffusers>=0.27.0->tensorrt-llm)
  Downloading importlib_metadata-8.7.0-py3-none-any.whl.metadata (4.8 kB)
Collecting huggingface-hub>=0.23.0 (from datasets==3.1.0->tensorrt-llm)
  Downloading huggingface_hub-0.34.4-py3-none-any.whl.metadata (14 kB)
Collecting protobuf>=4.25.1 (from onnx>=1.12.0->tensorrt-llm)
  Downloading protobuf-6.32.0-cp39-abi3-manylinux2014_x86_64.whl.metadata (593 bytes)
Collecting nvidia-ml-py>=12 (from tensorrt-llm)
  Downloading nvidia_ml_py-12.575.51-py3-none-any.whl.metadata (9.3 kB)
Requirement already satisfied: charset_normalizer<4,>=2 in /venv/main/lib/python3.12/site-packages (from requests>=2.32.2->datasets==3.1.0->tensorrt-llm) (3.4.2)
Requirement already satisfied: urllib3<3,>=1.21.1 in /venv/main/lib/python3.12/site-packages (from requests>=2.32.2->datasets==3.1.0->tensorrt-llm) (2.5.0)
Requirement already satisfied: certifi>=2017.4.17 in /venv/main/lib/python3.12/site-packages (from requests>=2.32.2->datasets==3.1.0->tensorrt-llm) (2025.6.15)
INFO: pip is looking at multiple versions of torchvision to determine which version is compatible with other requirements. This could take a while.
Collecting torchvision (from tensorrt-llm)
  Downloading torchvision-0.22.1-cp312-cp312-manylinux_2_28_x86_64.whl.metadata (6.1 kB)
Collecting pyproject_hooks (from build->tensorrt-llm)
  Downloading pyproject_hooks-1.2.0-py3-none-any.whl.metadata (1.3 kB)
Collecting cuda-bindings~=13.0.1 (from cuda-python->tensorrt-llm)
  Downloading cuda_bindings-13.0.1-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl.metadata (2.7 kB)
Collecting cuda-pathfinder~=1.1 (from cuda-python->tensorrt-llm)
  Downloading cuda_pathfinder-1.1.0-py3-none-any.whl.metadata (2.3 kB)
Collecting grpcio>=1.27.1 (from etcd3->tensorrt-llm)
  Downloading grpcio-1.74.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.8 kB)
Requirement already satisfied: six>=1.12.0 in /venv/main/lib/python3.12/site-packages (from etcd3->tensorrt-llm) (1.17.0)
Collecting tenacity>=6.1.0 (from etcd3->tensorrt-llm)
  Downloading tenacity-9.1.2-py3-none-any.whl.metadata (1.2 kB)
Collecting zipp>=3.20 (from importlib_metadata->diffusers>=0.27.0->tensorrt-llm)
  Downloading zipp-3.23.0-py3-none-any.whl.metadata (3.6 kB)
Collecting MarkupSafe>=2.0 (from jinja2->torch<=2.8.0a0,>=2.7.1->tensorrt-llm)
  Downloading MarkupSafe-3.0.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.0 kB)
Collecting contourpy>=1.0.1 (from matplotlib->tensorrt-llm)
  Downloading contourpy-1.3.3-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (5.5 kB)
Collecting cycler>=0.10 (from matplotlib->tensorrt-llm)
  Downloading cycler-0.12.1-py3-none-any.whl.metadata (3.8 kB)
Collecting fonttools>=4.22.0 (from matplotlib->tensorrt-llm)
  Downloading fonttools-4.59.1-cp312-cp312-manylinux1_x86_64.manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_5_x86_64.whl.metadata (108 kB)
Collecting kiwisolver>=1.3.1 (from matplotlib->tensorrt-llm)
  Downloading kiwisolver-1.4.9-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (6.3 kB)
Collecting pyparsing>=2.3.1 (from matplotlib->tensorrt-llm)
  Downloading pyparsing-3.2.3-py3-none-any.whl.metadata (5.0 kB)
Requirement already satisfied: python-dateutil>=2.7 in /venv/main/lib/python3.12/site-packages (from matplotlib->tensorrt-llm) (2.9.0.post0)
Collecting distro<2,>=1.7.0 (from openai->tensorrt-llm)
  Downloading distro-1.9.0-py3-none-any.whl.metadata (6.8 kB)
Collecting httpx<1,>=0.23.0 (from openai->tensorrt-llm)
  Downloading httpx-0.28.1-py3-none-any.whl.metadata (7.1 kB)
Collecting jiter<1,>=0.4.0 (from openai->tensorrt-llm)
  Downloading jiter-0.10.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (5.2 kB)
Collecting httpcore==1.* (from httpx<1,>=0.23.0->openai->tensorrt-llm)
  Downloading httpcore-1.0.9-py3-none-any.whl.metadata (21 kB)
INFO: pip is looking at multiple versions of opencv-python-headless to determine which version is compatible with other requirements. This could take a while.
Collecting opencv-python-headless (from tensorrt-llm)
  Downloading opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (20 kB)
Collecting pytz>=2020.1 (from pandas->tensorrt-llm)
  Downloading pytz-2025.2-py2.py3-none-any.whl.metadata (22 kB)
Collecting tzdata>=2022.7 (from pandas->tensorrt-llm)
  Downloading tzdata-2025.2-py2.py3-none-any.whl.metadata (1.4 kB)
Collecting markdown-it-py>=2.2.0 (from rich->nvidia-modelopt~=0.31.0->nvidia-modelopt[torch]~=0.31.0->tensorrt-llm)
  Downloading markdown_it_py-4.0.0-py3-none-any.whl.metadata (7.3 kB)
Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /venv/main/lib/python3.12/site-packages (from rich->nvidia-modelopt~=0.31.0->nvidia-modelopt[torch]~=0.31.0->tensorrt-llm) (2.19.2)
Collecting mdurl~=0.1 (from markdown-it-py>=2.2.0->rich->nvidia-modelopt~=0.31.0->nvidia-modelopt[torch]~=0.31.0->tensorrt-llm)
  Downloading mdurl-0.1.2-py3-none-any.whl.metadata (1.6 kB)
Downloading datasets-3.1.0-py3-none-any.whl (480 kB)
Downloading fastapi-0.115.4-py3-none-any.whl (94 kB)
Downloading h5py-3.12.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (5.4 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.4/5.4 MB 108.4 MB/s eta 0:00:00
Downloading llguidance-0.7.29-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (15.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 15.0/15.0 MB 43.8 MB/s eta 0:00:00
Downloading pillow-10.3.0-cp312-cp312-manylinux_2_28_x86_64.whl (4.5 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.5/4.5 MB 152.4 MB/s eta 0:00:00
Downloading triton-3.3.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (155.7 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 155.7/155.7 MB 121.5 MB/s eta 0:00:00
Downloading xgrammar-0.1.18-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (4.8 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.8/4.8 MB 13.4 MB/s eta 0:00:00
Downloading dill-0.3.8-py3-none-any.whl (116 kB)
Downloading fsspec-2024.9.0-py3-none-any.whl (179 kB)
Downloading multiprocess-0.70.16-py312-none-any.whl (146 kB)
Downloading numpy-1.26.4-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (18.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 18.0/18.0 MB 180.8 MB/s eta 0:00:00
Downloading pydantic-2.11.7-py3-none-any.whl (444 kB)
Downloading pydantic_core-2.33.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.0/2.0 MB 172.4 MB/s eta 0:00:00
Downloading setuptools-79.0.1-py3-none-any.whl (1.3 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.3/1.3 MB 176.6 MB/s eta 0:00:00
Downloading starlette-0.41.3-py3-none-any.whl (73 kB)
Downloading anyio-4.10.0-py3-none-any.whl (107 kB)
Downloading torch-2.7.1-cp312-cp312-manylinux_2_28_x86_64.whl (821.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 821.0/821.0 MB 100.8 MB/s eta 0:00:00
Downloading nvidia_cublas_cu12-12.6.4.1-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (393.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 393.1/393.1 MB 127.0 MB/s eta 0:00:00
Downloading nvidia_cuda_cupti_cu12-12.6.80-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (8.9 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.9/8.9 MB 82.8 MB/s eta 0:00:00
Downloading transformers-4.51.3-py3-none-any.whl (10.4 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 10.4/10.4 MB 129.6 MB/s eta 0:00:00
Downloading tokenizers-0.21.4-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.1/3.1 MB 161.3 MB/s eta 0:00:00
Downloading accelerate-1.10.0-py3-none-any.whl (374 kB)
Downloading aiohttp-3.12.15-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.7 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.7/1.7 MB 136.8 MB/s eta 0:00:00
Downloading multidict-6.6.4-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (256 kB)
Downloading yarl-1.20.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (355 kB)
Downloading aiohappyeyeballs-2.6.1-py3-none-any.whl (15 kB)
Downloading aiosignal-1.4.0-py3-none-any.whl (7.5 kB)
Downloading annotated_types-0.7.0-py3-none-any.whl (13 kB)
Downloading attrs-25.3.0-py3-none-any.whl (63 kB)
Downloading diffusers-0.35.1-py3-none-any.whl (4.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.1/4.1 MB 152.7 MB/s eta 0:00:00
Downloading huggingface_hub-0.34.4-py3-none-any.whl (561 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 561.5/561.5 kB 72.2 MB/s eta 0:00:00
Downloading frozenlist-1.7.0-cp312-cp312-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (241 kB)
Downloading h11-0.16.0-py3-none-any.whl (37 kB)
Downloading mpmath-1.3.0-py3-none-any.whl (536 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 536.2/536.2 kB 65.3 MB/s eta 0:00:00
Downloading onnx-1.18.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (17.6 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 17.6/17.6 MB 123.9 MB/s eta 0:00:00
Downloading propcache-0.3.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (224 kB)
Downloading protobuf-6.32.0-cp39-abi3-manylinux2014_x86_64.whl (322 kB)
Downloading pyarrow-21.0.0-cp312-cp312-manylinux_2_28_x86_64.whl (42.8 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 42.8/42.8 MB 146.7 MB/s eta 0:00:00
Downloading pynvml-12.0.0-py3-none-any.whl (26 kB)
Downloading nvidia_ml_py-12.575.51-py3-none-any.whl (47 kB)
Downloading regex-2025.7.34-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (801 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 801.9/801.9 kB 125.7 MB/s eta 0:00:00
Downloading safetensors-0.6.2-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (485 kB)
Downloading sentencepiece-0.2.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (1.4 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.4/1.4 MB 119.8 MB/s eta 0:00:00
Downloading sniffio-1.3.1-py3-none-any.whl (10 kB)
Downloading sympy-1.14.0-py3-none-any.whl (6.3 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.3/6.3 MB 175.0 MB/s eta 0:00:00
Downloading torchprofile-0.0.4-py3-none-any.whl (7.7 kB)
Downloading torchvision-0.22.1-cp312-cp312-manylinux_2_28_x86_64.whl (7.5 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7.5/7.5 MB 110.6 MB/s eta 0:00:00
Downloading typing_inspection-0.4.1-py3-none-any.whl (14 kB)
Downloading aenum-3.1.16-py3-none-any.whl (165 kB)
Downloading backoff-2.2.1-py3-none-any.whl (15 kB)
Downloading blake3-1.0.5-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (384 kB)
Downloading build-1.3.0-py3-none-any.whl (23 kB)
Downloading click-8.2.1-py3-none-any.whl (102 kB)
Downloading click_option_group-0.5.7-py3-none-any.whl (11 kB)
Downloading colored-2.3.1-py3-none-any.whl (19 kB)
Downloading cuda_python-13.0.1-py3-none-any.whl (7.6 kB)
Downloading cuda_bindings-13.0.1-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl (12.3 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 12.3/12.3 MB 46.0 MB/s eta 0:00:00
Downloading cuda_pathfinder-1.1.0-py3-none-any.whl (17 kB)
Downloading einops-0.8.1-py3-none-any.whl (64 kB)
Downloading grpcio-1.74.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (6.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.2/6.2 MB 168.2 MB/s eta 0:00:00
Downloading tenacity-9.1.2-py3-none-any.whl (28 kB)
Downloading evaluate-0.4.5-py3-none-any.whl (84 kB)
Downloading importlib_metadata-8.7.0-py3-none-any.whl (27 kB)
Downloading zipp-3.23.0-py3-none-any.whl (10 kB)
Downloading jinja2-3.1.6-py3-none-any.whl (134 kB)
Downloading MarkupSafe-3.0.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (23 kB)
Downloading lark-1.2.2-py3-none-any.whl (111 kB)
Downloading matplotlib-3.10.5-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (8.7 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.7/8.7 MB 172.5 MB/s eta 0:00:00
Downloading contourpy-1.3.3-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (362 kB)
Downloading cycler-0.12.1-py3-none-any.whl (8.3 kB)
Downloading fonttools-4.59.1-cp312-cp312-manylinux1_x86_64.manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_5_x86_64.whl (4.9 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.9/4.9 MB 151.3 MB/s eta 0:00:00
Downloading kiwisolver-1.4.9-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (1.5 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.5/1.5 MB 133.6 MB/s eta 0:00:00
Downloading pyparsing-3.2.3-py3-none-any.whl (111 kB)
Downloading meson-1.8.4-py3-none-any.whl (1.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.0/1.0 MB 169.2 MB/s eta 0:00:00
Downloading mpi4py-4.1.0-cp312-cp312-manylinux1_x86_64.manylinux_2_5_x86_64.whl (1.5 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.5/1.5 MB 10.1 MB/s eta 0:00:00
Downloading networkx-3.5-py3-none-any.whl (2.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.0/2.0 MB 147.2 MB/s eta 0:00:00
Downloading ninja-1.13.0-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (180 kB)
Downloading openai-1.101.0-py3-none-any.whl (810 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 810.8/810.8 kB 128.3 MB/s eta 0:00:00
Downloading distro-1.9.0-py3-none-any.whl (20 kB)
Downloading httpx-0.28.1-py3-none-any.whl (73 kB)
Downloading httpcore-1.0.9-py3-none-any.whl (78 kB)
Downloading jiter-0.10.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (352 kB)
Downloading opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (50.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 50.0/50.0 MB 157.7 MB/s eta 0:00:00
Downloading optimum-1.27.0-py3-none-any.whl (425 kB)
Downloading ordered_set-4.1.0-py3-none-any.whl (7.6 kB)
Downloading pandas-2.3.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (12.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 12.0/12.0 MB 108.8 MB/s eta 0:00:00
Downloading pytz-2025.2-py2.py3-none-any.whl (509 kB)
Downloading tzdata-2025.2-py2.py3-none-any.whl (347 kB)
Downloading peft-0.17.1-py3-none-any.whl (504 kB)
Downloading pulp-3.2.2-py3-none-any.whl (16.4 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 16.4/16.4 MB 42.5 MB/s eta 0:00:00
Downloading pyproject_hooks-1.2.0-py3-none-any.whl (10 kB)
Downloading rich-14.1.0-py3-none-any.whl (243 kB)
Downloading markdown_it_py-4.0.0-py3-none-any.whl (87 kB)
Downloading mdurl-0.1.2-py3-none-any.whl (10.0 kB)
Downloading scipy-1.16.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (35.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 35.2/35.2 MB 173.5 MB/s eta 0:00:00
Downloading StrEnum-0.4.15-py3-none-any.whl (8.9 kB)
Downloading tiktoken-0.11.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.2/1.2 MB 136.1 MB/s eta 0:00:00
Downloading uvicorn-0.35.0-py3-none-any.whl (66 kB)
Downloading xxhash-3.5.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (194 kB)
Building wheels for collected packages: flashinfer-python, tensorrt, tensorrt_cu12, etcd3
  Building wheel for flashinfer-python (pyproject.toml) ... done
  Created wheel for flashinfer-python: filename=flashinfer_python-0.2.5-py3-none-any.whl size=4124482 sha256=0c78ccfecc76c0cdea3d2ee4448141cdf47fb66e1aab42459a0d98523b2af86d
  Stored in directory: /root/.cache/pip/wheels/88/25/72/826d34ecab0d5e1a762f1762df1673f0cc029953de8744def3
  DEPRECATION: Building 'tensorrt' using the legacy setup.py bdist_wheel mechanism, which will be removed in a future version. pip 25.3 will enforce this behaviour change. A possible replacement is to use the standardized build interface by setting the `--use-pep517` option, (possibly combined with `--no-build-isolation`), or adding a `pyproject.toml` file to the source tree of 'tensorrt'. Discussion can be found at https://github.com/pypa/pip/issues/6334
  Building wheel for tensorrt (setup.py) ... done
  Created wheel for tensorrt: filename=tensorrt-10.11.0.33-py2.py3-none-any.whl size=46731 sha256=05f80505275b4f61f7bb3c13ec0465a0c3406981d8c7437e195b7cba199ea06e
  Stored in directory: /root/.cache/pip/wheels/7f/c3/c2/1e80a11ddcebc60a8c3bcde164db5e52dbf74dad28bf98603a
  DEPRECATION: Building 'tensorrt_cu12' using the legacy setup.py bdist_wheel mechanism, which will be removed in a future version. pip 25.3 will enforce this behaviour change. A possible replacement is to use the standardized build interface by setting the `--use-pep517` option, (possibly combined with `--no-build-isolation`), or adding a `pyproject.toml` file to the source tree of 'tensorrt_cu12'. Discussion can be found at https://github.com/pypa/pip/issues/6334
  Building wheel for tensorrt_cu12 (setup.py) ... done
  Created wheel for tensorrt_cu12: filename=tensorrt_cu12-10.11.0.33-py2.py3-none-any.whl size=17582 sha256=167d6b5a1c76241fee122185b9e76d62d6f672d1be77b4035497d2a85359192d
  Stored in directory: /root/.cache/pip/wheels/23/d4/e7/4e3c60d575978d9cac11a3f0e59f28419ed4e8a7cfcae388cb
  DEPRECATION: Building 'etcd3' using the legacy setup.py bdist_wheel mechanism, which will be removed in a future version. pip 25.3 will enforce this behaviour change. A possible replacement is to use the standardized build interface by setting the `--use-pep517` option, (possibly combined with `--no-build-isolation`), or adding a `pyproject.toml` file to the source tree of 'etcd3'. Discussion can be found at https://github.com/pypa/pip/issues/6334
  Building wheel for etcd3 (setup.py) ... done
  Created wheel for etcd3: filename=etcd3-0.12.0-py2.py3-none-any.whl size=42975 sha256=ad8cb450a436af1bf820c735f7651eda71f3611d8fa2ccf9e4802ee7e8b53bee
  Stored in directory: /root/.cache/pip/wheels/bb/ab/cb/8178a773ec0cee5434f923ad304941e794ed6a8392f0cd5f93
Successfully built flashinfer-python tensorrt tensorrt_cu12 etcd3
Installing collected packages: tensorrt_cu12_bindings, StrEnum, pytz, nvtx, nvidia-ml-py, nvidia-cusparselt-cu12, mpmath, blake3, aenum, zipp, xxhash, tzdata, typing-inspection, tenacity, sympy, sniffio, setuptools, sentencepiece, safetensors, regex, pyproject_hooks, pyparsing, pynvml, pydantic-core, pyarrow, pulp, protobuf, propcache, polygraphy, pillow, ordered-set, nvidia-nvtx-cu12, nvidia-nvjitlink-cu12, nvidia-nccl-cu12, nvidia-modelopt-core, nvidia-curand-cu12, nvidia-cufile-cu12, nvidia-cuda-runtime-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-cupti-cu12, nvidia-cublas-cu12, numpy, ninja, networkx, multidict, mpi4py, meson, mdurl, MarkupSafe, llguidance, lark, kiwisolver, jiter, h11, grpcio, fsspec, frozenlist, fonttools, einops, distro, dill, cycler, cuda-pathfinder, colored, click, backoff, attrs, annotated-types, aiohappyeyeballs, yarl, uvicorn, triton, tiktoken, tensorrt_cu12_libs, scipy, pydantic, pandas, opencv-python-headless, onnx, nvidia-cusparse-cu12, nvidia-cufft-cu12, nvidia-cudnn-cu12, multiprocess, markdown-it-py, jinja2, importlib_metadata, huggingface-hub, httpcore, h5py, etcd3, cuda-bindings, contourpy, click_option_group, build, anyio, aiosignal, tokenizers, tensorrt_cu12, starlette, rich, onnx_graphsurgeon, nvidia-cusolver-cu12, matplotlib, httpx, diffusers, cuda-python, aiohttp, transformers, torch, tensorrt, openai, nvidia-modelopt, fastapi, xgrammar, torchvision, optimum, flashinfer-python, datasets, accelerate, torchprofile, peft, evaluate, tensorrt-llm
  Attempting uninstall: setuptools
    Found existing installation: setuptools 80.9.0
    Uninstalling setuptools-80.9.0:
      Successfully uninstalled setuptools-80.9.0
  Attempting uninstall: fsspec
    Found existing installation: fsspec 2025.5.1
    Uninstalling fsspec-2025.5.1:
      Successfully uninstalled fsspec-2025.5.1
  Attempting uninstall: huggingface-hub
    Found existing installation: huggingface-hub 0.33.1
    Uninstalling huggingface-hub-0.33.1:
      Successfully uninstalled huggingface-hub-0.33.1
Successfully installed MarkupSafe-3.0.2 StrEnum-0.4.15 accelerate-1.10.0 aenum-3.1.16 aiohappyeyeballs-2.6.1 aiohttp-3.12.15 aiosignal-1.4.0 annotated-types-0.7.0 anyio-4.10.0 attrs-25.3.0 backoff-2.2.1 blake3-1.0.5 build-1.3.0 click-8.2.1 click_option_group-0.5.7 colored-2.3.1 contourpy-1.3.3 cuda-bindings-13.0.1 cuda-pathfinder-1.1.0 cuda-python-13.0.1 cycler-0.12.1 datasets-3.1.0 diffusers-0.35.1 dill-0.3.8 distro-1.9.0 einops-0.8.1 etcd3-0.12.0 evaluate-0.4.5 fastapi-0.115.4 flashinfer-python-0.2.5 fonttools-4.59.1 frozenlist-1.7.0 fsspec-2024.9.0 grpcio-1.74.0 h11-0.16.0 h5py-3.12.1 httpcore-1.0.9 httpx-0.28.1 huggingface-hub-0.34.4 importlib_metadata-8.7.0 jinja2-3.1.6 jiter-0.10.0 kiwisolver-1.4.9 lark-1.2.2 llguidance-0.7.29 markdown-it-py-4.0.0 matplotlib-3.10.5 mdurl-0.1.2 meson-1.8.4 mpi4py-4.1.0 mpmath-1.3.0 multidict-6.6.4 multiprocess-0.70.16 networkx-3.5 ninja-1.13.0 numpy-1.26.4 nvidia-cublas-cu12-12.6.4.1 nvidia-cuda-cupti-cu12-12.6.80 nvidia-cuda-nvrtc-cu12-12.6.77 nvidia-cuda-runtime-cu12-12.6.77 nvidia-cudnn-cu12-9.5.1.17 nvidia-cufft-cu12-11.3.0.4 nvidia-cufile-cu12-1.11.1.6 nvidia-curand-cu12-10.3.7.77 nvidia-cusolver-cu12-11.7.1.2 nvidia-cusparse-cu12-12.5.4.2 nvidia-cusparselt-cu12-0.6.3 nvidia-ml-py-12.575.51 nvidia-modelopt-0.31.0 nvidia-modelopt-core-0.31.0 nvidia-nccl-cu12-2.26.2 nvidia-nvjitlink-cu12-12.6.85 nvidia-nvtx-cu12-12.6.77 nvtx-0.2.13 onnx-1.18.0 onnx_graphsurgeon-0.5.8 openai-1.101.0 opencv-python-headless-4.11.0.86 optimum-1.27.0 ordered-set-4.1.0 pandas-2.3.2 peft-0.17.1 pillow-10.3.0 polygraphy-0.49.26 propcache-0.3.2 protobuf-6.32.0 pulp-3.2.2 pyarrow-21.0.0 pydantic-2.11.7 pydantic-core-2.33.2 pynvml-12.0.0 pyparsing-3.2.3 pyproject_hooks-1.2.0 pytz-2025.2 regex-2025.7.34 rich-14.1.0 safetensors-0.6.2 scipy-1.16.1 sentencepiece-0.2.1 setuptools-79.0.1 sniffio-1.3.1 starlette-0.41.3 sympy-1.14.0 tenacity-9.1.2 tensorrt-10.11.0.33 tensorrt-llm-0.21.0 tensorrt_cu12-10.11.0.33 tensorrt_cu12_bindings-10.11.0.33 tensorrt_cu12_libs-10.11.0.33 tiktoken-0.11.0 tokenizers-0.21.4 torch-2.7.1 torchprofile-0.0.4 torchvision-0.22.1 transformers-4.51.3 triton-3.3.1 typing-inspection-0.4.1 tzdata-2025.2 uvicorn-0.35.0 xgrammar-0.1.18 xxhash-3.5.0 yarl-1.20.1 zipp-3.23.0
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
'

```

```bash
Running the C++ executor examples:

(main) root@C.25238311:/workspace$ git clone https://github.com/NVIDIA/TensorRT-LLM.git
Cloning into 'TensorRT-LLM'...
remote: Enumerating objects: 110493, done.
remote: Counting objects: 100% (959/959), done.
remote: Compressing objects: 100% (713/713), done.
remote: Total 110493 (delta 659), reused 246 (delta 246), pack-reused 109534 (from 4)
Receiving objects: 100% (110493/110493), 1.56 GiB | 14.60 MiB/s, done.
Resolving deltas: 100% (77561/77561), done.
Updating files: 100% (6153/6153), done.
Filtering content: 100% (2409/2409), 1.65 GiB | 21.40 MiB/s, done.

```

```bash
1  python --version
2  pip3 --version
3  pip3 install tensorrt-llm --extra-index-url https://pypi.nvidia.com/
4  git clone https://github.com/NVIDIA/TensorRT-LLM.git
5  cd TensorRT-LLM
6  git lfs install
7  python3 ./scripts/build_wheel.py --benchmarks --cpp_only --clean
8  apt update
9  apt install -y libnuma-dev
10  pwd
11  python3 ./scripts/build_wheel.py     --cuda_architectures "90-real"     --benchmarks     --cpp_only     --clean
12  # Add NVIDIA's repository if not already added
13  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
14  dpkg -i cuda-keyring_1.1-1_all.deb
15  apt update
16  # Install TensorRT development libraries
17  apt install -y tensorrt-dev libnvinfer-dev libnvonnxparsers-dev libnvparsers-dev
18  python -c "import tensorrt as trt; print(trt.__file__)"
19  # Check what's available
20  find /venv/main/lib/python3.12/site-packages -name "*tensorrt*" -type d
21  ls -la /venv/main/lib/python3.12/site-packages/tensorrt*
22  python3 ./scripts/build_wheel.py     --cuda_architectures "90-real"     --benchmarks     --cpp_only     --clean     --tensorrt_root /venv/main/lib/python3.12/site-packages/tensorrt
23  python3 ./scripts/build_wheel.py     --cuda_architectures "90-real"     --benchmarks     --cpp_only     --clean     --trt_root /venv/main/lib/python3.12/site-packages/tensorrt_libs
24  tar -xzf TensorRT-10.11.0.Linux.x86_64-gnu.cuda-12.8.tar.gz
25  wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.11.0/tars/TensorRT-10.11.0.33.Linux.x86_64-gnu.cuda-12.9.tar.gz
26  ls
27  tar -xvf TensorRT-10.11.0.33.Linux.x86_64-gnu.cuda-12.9.tar.gz
28  ls
29  sudo mv TensorRT-10.11.0.33 /usr/local/tensorrt
30  python3 ./scripts/build_wheel.py     --cuda_architectures "90-real"     --benchmarks     --cpp_only     --clean     --trt_root /usr/local/tensorrt
31  mpirun --version
32  mpiexec --version
33  mpirun --version
34  mpiexec --version
35  python3 ./scripts/build_wheel.py     --cuda_architectures "90-real"     --benchmarks     --cpp_only     --clean     --trt_root /usr/local/tensorrt
36  history
```

