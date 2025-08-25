# LLM-Inference-Notes
LLM inference notes

## TensorRT-LLM

## Machines I have installed this on in the Cloud:
1. 2x H100 NVL
2. 1x RTX PRO 6000 WS
3. 1x H100 NVL
4. 1x H200 NVL - India
5. 1x RTX 5090
6. 1x H200 - France
7. 1x H100 SXM
 
<details>
  <summary>[RAW] Commands for installation:</summary>
  
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
  ```

</details>


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

## Checking Python version
```bash
python --version
```
### Output
```bash
Python 3.12.11
```

## Checking Pip version
```bash
pip3 --version
```
### Output
```bash
pip 25.1.1 from /venv/main/lib/python3.12/site-packages/pip (python 3.12)
```

## Pip installing TensorRT-LLM – Not required if install C++ libraries, takes 15 mins
```bash
pip3 install tensorrt-llm --extra-index-url https://pypi.nvidia.com/
```
### Output
```bash
Looking in indexes: https://pypi.org/simple, https://pypi.nvidia.com/
Collecting tensorrt-llm
  Downloading https://pypi.nvidia.com/tensorrt-llm/tensorrt_llm-0.21.0-cp312-cp312-linux_x86_64.whl (3932.9 MB)
     ━╺━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0.1/3.9 GB 15.5 MB/s eta 0:04:07
```

## Cloning TensorRT-LLM from Github
```bash
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git lfs install
```
### Output
```bash
(main) root@C.25296837:/venv/main/lib/python3.12/site-packages$ git clone https://github.com/NVIDIA/TensorRT-LLM.git
Cloning into 'TensorRT-LLM'...
remote: Enumerating objects: 110906, done.
remote: Counting objects: 100% (126/126), done.
remote: Compressing objects: 100% (94/94), done.
remote: Total 110906 (delta 58), reused 36 (delta 32), pack-reused 110780 (from 2)
Receiving objects: 100% (110906/110906), 1.55 GiB | 20.79 MiB/s, done.
Resolving deltas: 100% (78265/78265), done.
Updating files: 100% (6161/6161), done.
Filtering content:  24% (600/2409), 206.70 MiB | 4.59 MiB/s
```

## Installing C++ dependencies
```bash
python3 ./scripts/build_wheel.py --benchmarks --cpp_only --clean
apt update
apt install -y libnuma-dev
pwd
python3 ./scripts/build_wheel.py     --cuda_architectures "90-real"     --benchmarks     --cpp_only     --clean
# Add NVIDIA's repository if not already added
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt update
# Install TensorRT development libraries
apt install -y tensorrt-dev libnvinfer-dev libnvonnxparsers-dev libnvparsers-dev
```

```
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


