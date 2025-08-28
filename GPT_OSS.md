Running and PostTraining GPT OSS

https://openai.com/index/introducing-gpt-oss/

# Trying with AMD machine
AMD MI300X
1 GPU - 192 GB VRAM - 20 vCPU - 240 GB RAM
Boot disk: 720 GB NVMe- Scratch disk: 5 TB NVMe
vLLM 0.8.6 on Ubuntu 24.04

# GPT OSS 120B
openai/gpt-oss-120b
Our larger full-sized model
Best with ≥60GB VRAM
Can fit on a single H100 or multi-GPU setups

# VLLM and GPT-OSS-120b
https://cookbook.openai.com/articles/gpt-oss/run-vllm

```
docker exec -it rocm /bin/bash
```
docker exec: This command executes a new process inside an already running container.
-it: This is a combination of two flags:
-i (--interactive): Keeps the standard input (STDIN) open, allowing interaction with the shell and the typing of commands.
-t (--tty): Allocates a pseudo-terminal, which makes the shell session look and behave like a normal terminal.
rocm: This is the name of the target container where the command will be run. "ROCm" likely refers to a container with AMD's ROCm software stack, used for GPU-accelerated computing.
/bin/bash: This is the command to execute inside the container, in this case, starting a Bash shell. 
