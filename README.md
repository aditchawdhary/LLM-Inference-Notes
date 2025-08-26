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

### Checking Python version
```bash
python --version
```
Output
```bash
Python 3.12.11
```
### Checking Ubuntu version
```
lsb_release -a
```
Output
```
No LSB modules are available.
Distributor ID: Ubuntu
Description:    Ubuntu 24.04.2 LTS
Release:        24.04
Codename:       noble
```

<details>
  <summary>## Checking Pip version</summary>
  
  ```bash
   pip3 --version
   ```
   Output
   ```bash
   pip 25.1.1 from /venv/main/lib/python3.12/site-packages/pip (python 3.12)
   ```
   
   ## Pip installing TensorRT-LLM – Not required if install C++ libraries, takes 15 mins
   ```bash
   pip3 install tensorrt-llm --extra-index-url https://pypi.nvidia.com/
   ```
   Output
   ```bash
   Looking in indexes: https://pypi.org/simple, https://pypi.nvidia.com/
   Collecting tensorrt-llm
     Downloading https://pypi.nvidia.com/tensorrt-llm/tensorrt_llm-0.21.0-cp312-cp312-linux_x86_64.whl (3932.9 MB)
        ━╺━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0.1/3.9 GB 15.5 MB/s eta 0:04:07
  ```
</details>

## Check CUDA version
```bash
nvcc --version
```
Output
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Tue_Oct_29_23:50:19_PDT_2024
Cuda compilation tools, release 12.6, V12.6.85
Build cuda_12.6.r12.6/compiler.35059454_0
```

## Cloning TensorRT-LLM from Github
```bash
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git lfs install
```
Output
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
Installing MPI
```bash
sudo apt install openmpi-bin openmpi-common libopenmpi-dev
mpirun --version
mpiexec --version
```
Installing TensorRT, this can take upto 30 mins
```bash
wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.11.0/tars/TensorRT-10.11.0.33.Linux.x86_64-gnu.cuda-12.9.tar.gz
tar -xvf TensorRT-10.11.0.33.Linux.x86_64-gnu.cuda-12.9.tar.gz
```
Finding NCCL
```bash
find /usr /opt -name "libnccl.so*" 2>/dev/null
```
Output
```bash
/usr/lib/x86_64-linux-gnu/libnccl.so
/usr/lib/x86_64-linux-gnu/libnccl.so.2
/usr/lib/x86_64-linux-gnu/libnccl.so.2.23.4
```


## Running build_wheel.py
Make sure, trt_root is the directory which contains the files inside TRT not the directory which has the TRT directory, takes upto 1 hour
```bash
python3 ./scripts/build_wheel.py --benchmarks --cpp_only \
  --clean --trt_root=/workspace/TensorRT-LLM/TensorRT-10.11.0.33 \
  --nccl_root=/usr/lib/x86_64-linux-gnu/
```

Another command for H100 config, and faster build
```bash
python3 ./scripts/build_wheel.py --cpp_only --clean --fast_build \
  --cuda_architectures="native" \
  --trt_root=/workspace/TensorRT-LLM/TensorRT-10.11.0.33 \
  --nccl_root=/usr/lib/x86_64-linux-gnu/
```
