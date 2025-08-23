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

> Activated conda/uv virtual environment at /venv/main
(main) root@C.25238311:/workspace$ python --version
> Python 3.12.11
(main) root@C.25238311:/workspace$ pip3 --version
> pip 25.1.1 from /venv/main/lib/python3.12/site-packages/pip (python 3.12)
(main) root@C.25238311:/workspace$ pip3 install tensorrt-llm --extra-index-url https://pypi.nvidia.com/
> Looking in indexes: https://pypi.org/simple, https://pypi.nvidia.com/
> Collecting tensorrt-llm
>  Downloading https://pypi.nvidia.com/tensorrt-llm/tensorrt_llm-0.21.0-cp312-cp312-linux_x86_64.whl (3932.9 MB)
```
