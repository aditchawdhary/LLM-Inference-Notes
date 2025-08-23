# LLM-Inference-Notes
LLM inference notes

## TensorRT-LLM
```bash

# Machines I have installed this on in the Cloud:
# 2x H100 NVL
# 1x RTX PRO 6000 WS
# 1x H100 NVL


# Commands for installation:
pip3 install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

sudo apt-get -y install libopenmpi-dev
root@C.25237343:/$ pip3 install --extra-index-url https://pypi.nvidia.com/ tensorrt-llm
# TensorRT-LLM 0.21.0 doesn't have a pre-built wheel for Python 3.11.13.
root@C.25237925:/workspace$ python3.10 -m pip install --extra-index-url https://pypi.nvidia.com/ tensorrt-llm==0.21.0
```
