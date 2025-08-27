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
8. 2x A100 SXM4 with 80GB VRAM each
 
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


OUTPUT REACHED:
```
[ 54%] Built target _context_attention_kernels_90
[ 54%] Linking CUDA device code CMakeFiles/cutlass_src.dir/cmake_device_link.o
[ 54%] Linking CXX static library libcutlass_src.a
[ 54%] Built target cutlass_src
[ 54%] Linking CUDA device code CMakeFiles/low_latency_gemm_src.dir/cmake_device_link.o
[ 54%] Linking CUDA static library liblow_latency_gemm_src.a
[ 54%] Built target low_latency_gemm_src
[ 54%] Built target selective_scan_src
[ 54%] Built target _moe_gemm_src_instantiations_80
```

## VAST.AI
```
vastai create instance <OFFER_ID> --image nvcr.io/nvidia/tensorrt-llm/release:latest --env \
'-p 8888:8888 -p 22:22 -e NVIDIA_VISIBLE_DEVICES=all -e CUDA_VISIBLE_DEVICES=all -e \
DOCKER_HOST=tcp://localhost:2376 --ipc=host --shm-size=32g --ulimit memlock=-1 --ulimit \
 stack=67108864' --onstart-cmd 'entrypoint.sh;;#!/bin/bash;# Update system and install \
essentials;apt-get update && apt-get install -y htop vim git curl wget;;# Install Docker \
if not present;if ! command -v docker &> /dev/null; then curl -fsSL https://get.docker.com \
-o get-docker.sh;sh get-docker.sh;rm get-docker.sh;fi;;# Start Docker daemon in background; \
service docker start;dockerd --host=unix:///var/run/docker.sock --host=tcp://0.0.0.0:2376 &;; \
# Wait for Docker daemon to be ready;sleep 5;while ! docker info &> /dev/null; do;echo "Waiting \
for Docker daemon to start...";sleep 2;done;;echo "Docker daemon is running";;# Install Jupyter \
if not present;pip install jupyter notebook ipywidgets matplotlib;;# Configure Jupyter;mkdir -p \
 ~/.jupyter;cat > ~/.jupyter/jupyter_notebook_config.py << EOF;c.NotebookApp.ip = '\''0.0.0.0'\''; \
c.NotebookApp.port = 8888;c.NotebookApp.allow_root = True;c.NotebookApp.token = '\'''\''; \
c.NotebookApp.password = '\'''\'';EOF;;# Start Jupyter in background;nohup jupyter notebook \
--allow-root --no-browser &;;# Keep container running - THIS IS CRITICAL;tail -f /dev/null' \
--disk 80 --jupyter --ssh --direct
```
<img width="175" height="200" alt="image" src="https://github.com/user-attachments/assets/578e4ac5-5873-49cd-97ed-6ed7369e38a2" />
<img width="175" height="235" alt="image" src="https://github.com/user-attachments/assets/450a41d0-9a12-426a-ba80-348e0ff09557" />


🔥🔥🔥🔥🔥🔥

### Build engine optimized for dual A100s with maximum throughput
🚀 Build High-Performance Engine for Stress Testing:
```
trtllm-build \
    --checkpoint_dir /workspace/test_model/checkpoint \
    --output_dir /workspace/test_model/engine_stress \
    --gemm_plugin float16 \
    --gpt_attention_plugin float16 \
    --max_batch_size 64 \
    --max_input_len 2048 \
    --max_seq_len 4096 \
    --max_beam_width 1 \
    --max_num_tokens 32768 \
    --use_fused_mlp enable \
    --context_fmha enable \
    --remove_input_padding enable
```
<details>
  <summary>RAW Output</summary>
  
  ```bash
  2025-08-27 08:22:19,541 - INFO - flashinfer.jit: Prebuilt kernels not found, using JIT backend
  [TensorRT-LLM] TensorRT-LLM version: 0.20.0
  [08/27/2025-08:22:19] [TRT-LLM] [I] Set bert_attention_plugin to auto.
  [08/27/2025-08:22:19] [TRT-LLM] [I] Set gpt_attention_plugin to float16.
  [08/27/2025-08:22:19] [TRT-LLM] [I] Set gemm_plugin to float16.
  [08/27/2025-08:22:19] [TRT-LLM] [I] Set gemm_swiglu_plugin to None.
  [08/27/2025-08:22:19] [TRT-LLM] [I] Set fp8_rowwise_gemm_plugin to None.
  [08/27/2025-08:22:19] [TRT-LLM] [I] Set nccl_plugin to auto.
  [08/27/2025-08:22:19] [TRT-LLM] [I] Set lora_plugin to None.
  [08/27/2025-08:22:19] [TRT-LLM] [I] Set dora_plugin to False.
  [08/27/2025-08:22:19] [TRT-LLM] [I] Set moe_plugin to auto.
  [08/27/2025-08:22:19] [TRT-LLM] [I] Set mamba_conv1d_plugin to auto.
  [08/27/2025-08:22:19] [TRT-LLM] [I] Set low_latency_gemm_plugin to None.
  [08/27/2025-08:22:19] [TRT-LLM] [I] Set low_latency_gemm_swiglu_plugin to None.
  [08/27/2025-08:22:19] [TRT-LLM] [I] Set gemm_allreduce_plugin to None.
  [08/27/2025-08:22:19] [TRT-LLM] [I] Set context_fmha to True.
  [08/27/2025-08:22:19] [TRT-LLM] [I] Set bert_context_fmha_fp32_acc to False.
  [08/27/2025-08:22:19] [TRT-LLM] [I] Set remove_input_padding to True.
  [08/27/2025-08:22:19] [TRT-LLM] [I] Set norm_quant_fusion to False.
  [08/27/2025-08:22:19] [TRT-LLM] [I] Set reduce_fusion to False.
  [08/27/2025-08:22:19] [TRT-LLM] [I] Set user_buffer to False.
  [08/27/2025-08:22:19] [TRT-LLM] [I] Set tokens_per_block to 32.
  [08/27/2025-08:22:19] [TRT-LLM] [I] Set use_paged_context_fmha to True.
  [08/27/2025-08:22:19] [TRT-LLM] [I] Set use_fp8_context_fmha to True.
  [08/27/2025-08:22:19] [TRT-LLM] [I] Set fuse_fp4_quant to False.
  [08/27/2025-08:22:19] [TRT-LLM] [I] Set multiple_profiles to False.
  [08/27/2025-08:22:19] [TRT-LLM] [I] Set paged_state to True.
  [08/27/2025-08:22:19] [TRT-LLM] [I] Set streamingllm to False.
  [08/27/2025-08:22:19] [TRT-LLM] [I] Set use_fused_mlp to True.
  [08/27/2025-08:22:19] [TRT-LLM] [I] Set pp_reduce_scatter to False.
  [08/27/2025-08:22:19] [TRT-LLM] [I] Set dtype to float16.
  [08/27/2025-08:22:19] [TRT-LLM] [I] Set paged_kv_cache to True.
  [08/27/2025-08:22:19] [TRT-LLM] [W] Overriding paged_state to False
  [08/27/2025-08:22:19] [TRT-LLM] [I] Set paged_state to False.
  [08/27/2025-08:22:19] [TRT-LLM] [W] max_seq_len 4096 is larger than max_position_embeddings 1024 * rotary scaling 1, the model accuracy might be affected
  [08/27/2025-08:22:19] [TRT-LLM] [W] remove_input_padding is enabled, while opt_num_tokens is not set, setting to max_batch_size*max_beam_width. 
  [08/27/2025-08:22:19] [TRT-LLM] [W] Specifying a `max_num_tokens` larger than 16384 is usually not recommended, we do not expect perf gain with that and too large `max_num_tokens` could possibly exceed the TensorRT tensor volume, causing runtime errors. Got `max_num_tokens` = 32768
  [08/27/2025-08:22:19] [TRT-LLM] [W] padding removal and fMHA are both enabled, max_input_len is not required and will be ignored
  [08/27/2025-08:22:19] [TRT-LLM] [I] Set use_fp8_context_fmha to False.
  [08/27/2025-08:22:19] [TRT-LLM] [W] FP8 Context FMHA is disabled because it must be used together with the fp8 quantization workflow.
  [08/27/2025-08:22:20] [TRT] [I] [MemUsageChange] Init CUDA: CPU +26, GPU +0, now: CPU 296, GPU 424 (MiB)
  [08/27/2025-08:22:21] [TRT] [I] [MemUsageChange] Init builder kernel library: CPU +1639, GPU +8, now: CPU 2136, GPU 432 (MiB)
  [08/27/2025-08:22:21] [TRT-LLM] [I] Set nccl_plugin to None.
  [08/27/2025-08:22:22] [TRT-LLM] [I] Total time of constructing network from module object 2.3651156425476074 seconds
  [08/27/2025-08:22:22] [TRT-LLM] [I] Total optimization profiles added: 1
  [08/27/2025-08:22:22] [TRT-LLM] [I] Total time to initialize the weights in network Unnamed Network 0: 00:00:00
  [08/27/2025-08:22:22] [TRT-LLM] [I] Build TensorRT engine Unnamed Network 0
  [08/27/2025-08:22:25] [TRT] [I] Global timing cache in use. Profiling results in this builder pass will be stored.
  [08/27/2025-08:22:25] [TRT] [I] Compiler backend is used during engine build.
  [08/27/2025-08:22:28] [TRT] [I] Detected 17 inputs and 1 output network tensors.
  [08/27/2025-08:22:30] [TRT] [I] Total Host Persistent Memory: 36544 bytes
  [08/27/2025-08:22:30] [TRT] [I] Total Device Persistent Memory: 0 bytes
  [08/27/2025-08:22:30] [TRT] [I] Max Scratch Memory: 105548288 bytes
  [08/27/2025-08:22:30] [TRT] [I] [BlockAssignment] Started assigning block shifts. This will take 210 steps to complete.
  [08/27/2025-08:22:30] [TRT] [I] [BlockAssignment] Algorithm ShiftNTopDown took 10.6115ms to assign 18 blocks to 210 nodes requiring 503421440 bytes.
  [08/27/2025-08:22:30] [TRT] [I] Total Activation Memory: 503420928 bytes
  [08/27/2025-08:22:30] [TRT] [I] Total Weights Memory: 326085504 bytes
  [08/27/2025-08:22:30] [TRT] [I] Compiler backend is used during engine execution.
  [08/27/2025-08:22:30] [TRT] [I] Engine generation completed in 5.37531 seconds.
  [08/27/2025-08:22:30] [TRT] [I] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 0 MiB, GPU 385 MiB
  [08/27/2025-08:22:30] [TRT-LLM] [I] Total time of building Unnamed Network 0: 00:00:08
  [08/27/2025-08:22:30] [TRT] [I] Serialized 27 bytes of code generator cache.
  [08/27/2025-08:22:30] [TRT] [I] Serialized 153518 bytes of compilation cache.
  [08/27/2025-08:22:30] [TRT] [I] Serialized 12 timing cache entries
  [08/27/2025-08:22:30] [TRT-LLM] [I] Timing cache serialized to model.cache
  [08/27/2025-08:22:30] [TRT-LLM] [I] Build phase peak memory: 6978.95 MB, children: 12049.21 MB
  [08/27/2025-08:22:31] [TRT-LLM] [I] Serializing engine to /workspace/test_model/engine_stress/rank0.engine...
  [08/27/2025-08:22:31] [TRT-LLM] [I] Engine serialized. Total time: 00:00:00
  [08/27/2025-08:22:31] [TRT-LLM] [I] Total time of building all engines: 00:00:11
  ```
</details>

```
trtllm-build \
    --checkpoint_dir /workspace/test_model/checkpoint \
    --output_dir /workspace/test_model/engine_dual_gpu \
    --gemm_plugin float16 \
    --gpt_attention_plugin float16 \
    --max_batch_size 128 \
    --max_input_len 2048 \
    --max_seq_len 4096 \
    --max_beam_width 1 \
    --max_num_tokens 65536 \
    --use_fused_mlp enable \
    --context_fmha enable \
    --remove_input_padding enable \
    --workers 2
```
<details>
<summary>RAW OUTPUT:</summary>

```bash
2025-08-27 09:13:48,052 - INFO - flashinfer.jit: Prebuilt kernels not found, using JIT backend
[TensorRT-LLM] TensorRT-LLM version: 0.20.0
[08/27/2025-09:13:48] [TRT-LLM] [I] Set bert_attention_plugin to auto.
[08/27/2025-09:13:48] [TRT-LLM] [I] Set gpt_attention_plugin to float16.
[08/27/2025-09:13:48] [TRT-LLM] [I] Set gemm_plugin to float16.
[08/27/2025-09:13:48] [TRT-LLM] [I] Set gemm_swiglu_plugin to None.
[08/27/2025-09:13:48] [TRT-LLM] [I] Set fp8_rowwise_gemm_plugin to None.
[08/27/2025-09:13:48] [TRT-LLM] [I] Set nccl_plugin to auto.
[08/27/2025-09:13:48] [TRT-LLM] [I] Set lora_plugin to None.
[08/27/2025-09:13:48] [TRT-LLM] [I] Set dora_plugin to False.
[08/27/2025-09:13:48] [TRT-LLM] [I] Set moe_plugin to auto.
[08/27/2025-09:13:48] [TRT-LLM] [I] Set mamba_conv1d_plugin to auto.
[08/27/2025-09:13:48] [TRT-LLM] [I] Set low_latency_gemm_plugin to None.
[08/27/2025-09:13:48] [TRT-LLM] [I] Set low_latency_gemm_swiglu_plugin to None.
[08/27/2025-09:13:48] [TRT-LLM] [I] Set gemm_allreduce_plugin to None.
[08/27/2025-09:13:48] [TRT-LLM] [I] Set context_fmha to True.
[08/27/2025-09:13:48] [TRT-LLM] [I] Set bert_context_fmha_fp32_acc to False.
[08/27/2025-09:13:48] [TRT-LLM] [I] Set remove_input_padding to True.
[08/27/2025-09:13:48] [TRT-LLM] [I] Set norm_quant_fusion to False.
[08/27/2025-09:13:48] [TRT-LLM] [I] Set reduce_fusion to False.
[08/27/2025-09:13:48] [TRT-LLM] [I] Set user_buffer to False.
[08/27/2025-09:13:48] [TRT-LLM] [I] Set tokens_per_block to 32.
[08/27/2025-09:13:48] [TRT-LLM] [I] Set use_paged_context_fmha to True.
[08/27/2025-09:13:48] [TRT-LLM] [I] Set use_fp8_context_fmha to True.
[08/27/2025-09:13:48] [TRT-LLM] [I] Set fuse_fp4_quant to False.
[08/27/2025-09:13:48] [TRT-LLM] [I] Set multiple_profiles to False.
[08/27/2025-09:13:48] [TRT-LLM] [I] Set paged_state to True.
[08/27/2025-09:13:48] [TRT-LLM] [I] Set streamingllm to False.
[08/27/2025-09:13:48] [TRT-LLM] [I] Set use_fused_mlp to True.
[08/27/2025-09:13:48] [TRT-LLM] [I] Set pp_reduce_scatter to False.
2025-08-27 09:13:59,741 - INFO - flashinfer.jit: Prebuilt kernels not found, using JIT backend
[TensorRT-LLM] TensorRT-LLM version: 0.20.0
[08/27/2025-09:13:59] [TRT-LLM] [I] Set dtype to float16.
[08/27/2025-09:13:59] [TRT-LLM] [I] Set paged_kv_cache to True.
[08/27/2025-09:13:59] [TRT-LLM] [W] Overriding paged_state to False
[08/27/2025-09:13:59] [TRT-LLM] [I] Set paged_state to False.
[08/27/2025-09:13:59] [TRT-LLM] [W] max_seq_len 4096 is larger than max_position_embeddings 1024 * rotary scaling 1, the model accuracy might be affected
[08/27/2025-09:13:59] [TRT-LLM] [W] remove_input_padding is enabled, while opt_num_tokens is not set, setting to max_batch_size*max_beam_width. 
[08/27/2025-09:13:59] [TRT-LLM] [W] Specifying a `max_num_tokens` larger than 16384 is usually not recommended, we do not expect perf gain with that and too large `max_num_tokens` could possibly exceed the TensorRT tensor volume, causing runtime errors. Got `max_num_tokens` = 65536
[08/27/2025-09:13:59] [TRT-LLM] [W] padding removal and fMHA are both enabled, max_input_len is not required and will be ignored
[08/27/2025-09:13:59] [TRT-LLM] [I] Set use_fp8_context_fmha to False.
[08/27/2025-09:13:59] [TRT-LLM] [W] FP8 Context FMHA is disabled because it must be used together with the fp8 quantization workflow.
[08/27/2025-09:14:00] [TRT] [I] [MemUsageChange] Init CUDA: CPU +26, GPU +0, now: CPU 296, GPU 424 (MiB)
[08/27/2025-09:14:02] [TRT] [I] [MemUsageChange] Init builder kernel library: CPU +1635, GPU +8, now: CPU 2132, GPU 432 (MiB)
[08/27/2025-09:14:02] [TRT-LLM] [I] Set nccl_plugin to None.
[08/27/2025-09:14:03] [TRT-LLM] [I] Total time of constructing network from module object 3.822653293609619 seconds
[08/27/2025-09:14:03] [TRT-LLM] [I] Total optimization profiles added: 1
[08/27/2025-09:14:03] [TRT-LLM] [I] Total time to initialize the weights in network Unnamed Network 0: 00:00:00
[08/27/2025-09:14:03] [TRT-LLM] [I] Build TensorRT engine Unnamed Network 0
[08/27/2025-09:14:10] [TRT] [I] Global timing cache in use. Profiling results in this builder pass will be stored.
[08/27/2025-09:14:10] [TRT] [I] Compiler backend is used during engine build.
[08/27/2025-09:14:16] [TRT] [I] Detected 17 inputs and 1 output network tensors.
[08/27/2025-09:14:19] [TRT] [I] Total Host Persistent Memory: 36544 bytes
[08/27/2025-09:14:19] [TRT] [I] Total Device Persistent Memory: 0 bytes
[08/27/2025-09:14:19] [TRT] [I] Max Scratch Memory: 206474752 bytes
[08/27/2025-09:14:19] [TRT] [I] [BlockAssignment] Started assigning block shifts. This will take 210 steps to complete.
[08/27/2025-09:14:19] [TRT] [I] [BlockAssignment] Algorithm ShiftNTopDown took 10.7952ms to assign 18 blocks to 210 nodes requiring 1006836224 bytes.
[08/27/2025-09:14:19] [TRT] [I] Total Activation Memory: 1006835712 bytes
[08/27/2025-09:14:20] [TRT] [I] Total Weights Memory: 326085504 bytes
[08/27/2025-09:14:20] [TRT] [I] Compiler backend is used during engine execution.
[08/27/2025-09:14:20] [TRT] [I] Engine generation completed in 9.17555 seconds.
[08/27/2025-09:14:20] [TRT] [I] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 0 MiB, GPU 769 MiB
[08/27/2025-09:14:20] [TRT-LLM] [I] Total time of building Unnamed Network 0: 00:00:16
[08/27/2025-09:14:20] [TRT] [I] Serialized 27 bytes of code generator cache.
[08/27/2025-09:14:20] [TRT] [I] Serialized 153132 bytes of compilation cache.
[08/27/2025-09:14:20] [TRT] [I] Serialized 12 timing cache entries
[08/27/2025-09:14:20] [TRT-LLM] [I] Timing cache serialized to model.cache
[08/27/2025-09:14:20] [TRT-LLM] [I] Build phase peak memory: 6977.16 MB, children: 12047.94 MB
[08/27/2025-09:14:20] [TRT-LLM] [I] Serializing engine to /workspace/test_model/engine_dual_gpu/rank0.engine...
[08/27/2025-09:14:21] [TRT-LLM] [I] Engine serialized. Total time: 00:00:00
[08/27/2025-09:14:26] [TRT-LLM] [I] Total time of building all engines: 00:00:37
```
</details>

### Create large dataset for stress testing
📊 Create Massive Stress Test Dataset:
```
python prepare_dataset.py \
    --tokenizer gpt2 \
    --output /workspace/stress_dataset.json \
    token-norm-dist \
    --num-requests 10000 \
    --input-mean 1024 \
    --input-stdev 256 \
    --output-mean 512 \
    --output-stdev 128
```
### Push maximum concurrent requests
1. Maximum Throughput Test:
🔥 Extreme Stress Tests:

```
/app/tensorrt_llm/benchmarks/cpp/gptManagerBenchmark \
    --engine_dir /workspace/test_model/engine_stress \
    --type inflight \
    --dataset /workspace/stress_dataset.json \
    --max_num_samples 5000 \
    --request_rate 200.0 \
    --concurrency 64 \
    --log_level info \
    --log_iteration_data
```

### Test with maximum batch size and long sequences
2. Memory Pressure Test:
```
/app/tensorrt_llm/benchmarks/cpp/gptManagerBenchmark \
    --engine_dir /workspace/test_model/engine_stress \
    --type inflight \
    --dataset /workspace/stress_dataset.json \
    --max_num_samples 10000 \
    --max_batch_size 64 \
    --enable_batch_size_tuning \
    --log_level info
```

### Long-running sustained load
3. Sustained Load Test:
```
/app/tensorrt_llm/benchmarks/cpp/gptManagerBenchmark \
    --engine_dir /workspace/test_model/engine_stress \
    --type inflight \
    --dataset /workspace/stress_dataset.json \
    --max_num_samples 20000 \
    --request_rate 100.0 \
    --concurrency 32 \
    --warm_up 10
```

### Monitor GPU utilization during tests
🌡️ Monitor GPU Usage:
```
watch -n 1 nvidia-smi
```

### Check if we can use both GPUs
⚡ Multi-GPU Test (if supported):
```
CUDA_VISIBLE_DEVICES=0,1 /app/tensorrt_llm/benchmarks/cpp/gptManagerBenchmark \
    --engine_dir /workspace/test_model/engine_stress \
    --type inflight \
    --dataset /workspace/stress_dataset.json \
    --max_num_samples 5000 \
    --request_rate 300.0 \
    --concurrency 128
```
Let's start with building the high-performance engine first. This will take advantage of your dual A100s and really push the batch manager to its limits!

The goal is to:

Max out GPU utilization (get those temps up! 🌡️)
Test batch manager scalability with high concurrency
Push memory limits with large batches
Measure peak throughput your hardware can achieve

```
# If you want MAXIMUM; 💥 Nuclear Option - 4 Instances (2 per GPU):
for gpu in 0 1; do
    for instance in 1 2; do
        CUDA_VISIBLE_DEVICES=$gpu /app/tensorrt_llm/benchmarks/cpp/gptManagerBenchmark \
            --engine_dir /workspace/test_model/engine_stress \
            --type inflight \
            --dataset /workspace/stress_dataset.json \
            --max_num_samples 2500 \
            --request_rate 150.0 \
            --log_level error &
    done
done
```
<img width="900" height="306" alt="image" src="https://github.com/user-attachments/assets/932828bf-5755-4f2e-82d9-a16deff3c96d" />

OUTPUT:
[BENCHMARK] num_samples 2500
[BENCHMARK] num_error_samples 0


[BENCHMARK] num_samples 2500
[BENCHMARK] total_latency(ms) 81970.45
[BENCHMARK] seq_throughput(seq/sec) 30.50
[BENCHMARK] token_throughput(token/sec) 15672.87
[BENCHMARK] avg_acceptance_rate(tokens/decoding steps) 1.00


[BENCHMARK] avg_sequence_latency(ms) 34195.29
[BENCHMARK] max_sequence_latency(ms) 66902.80
[BENCHMARK] min_sequence_latency(ms) 778.81
[BENCHMARK] p99_sequence_latency(ms) 66332.86
[BENCHMARK] p90_sequence_latency(ms) 60695.08
[BENCHMARK] p50_sequence_latency(ms) 34129.66


[BENCHMARK] num_samples 2500
[BENCHMARK] num_error_samples 0


[BENCHMARK] num_samples 2500
[BENCHMARK] total_latency(ms) 184821.02
[BENCHMARK] seq_throughput(seq/sec) 13.53
[BENCHMARK] token_throughput(token/sec) 6951.11
[BENCHMARK] avg_acceptance_rate(tokens/decoding steps) 1.00


[BENCHMARK] avg_sequence_latency(ms) 86171.27
[BENCHMARK] max_sequence_latency(ms) 169654.69
[BENCHMARK] min_sequence_latency(ms) 1822.40
[BENCHMARK] p99_sequence_latency(ms) 167939.34
[BENCHMARK] p90_sequence_latency(ms) 153332.48
[BENCHMARK] p50_sequence_latency(ms) 86028.87

[BENCHMARK] num_samples 2500
[BENCHMARK] num_error_samples 0


[BENCHMARK] num_samples 2500
[BENCHMARK] total_latency(ms) 192773.70

[BENCHMARK] seq_throughput(seq/sec) 12.97
[BENCHMARK] token_throughput(token/sec) 6664.35
[BENCHMARK] avg_acceptance_rate(tokens/decoding steps) 1.00
[BENCHMARK] avg_sequence_latency(ms) 95497.55
[BENCHMARK] max_sequence_latency(ms) 177765.50
[BENCHMARK] min_sequence_latency(ms) 1830.76
[BENCHMARK] p99_sequence_latency(ms) 177170.33
[BENCHMARK] p90_sequence_latency(ms) 170575.42
[BENCHMARK] p50_sequence_latency(ms) 96085.57
