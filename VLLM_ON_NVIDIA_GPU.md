# Running VLLM on Vast.AI

```
conda create -n vllm-phi4flash-test python=3.10 -y
conda activate vllm-phi4flash-test
```

```
# Clone the main vLLM repository
git clone https://github.com/vllm-project/vllm.git
cd vllm

# Fetch the specific PR branch from aditchawdhary's fork
git fetch origin pull/23996/head:pr-23996-phi4flash
git checkout pr-23996-phi4flash

# Verify you're on the correct branch and see the changes
git branch -v
git log --oneline -n 5
git show --name-only HEAD  # Show files changed in latest commit
```

```
pip3 install -r requirements/build.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
VLLM_USE_PRECOMPILED=1 pip install -e . --verbose 
```

NOT 
```
VLLM_USE_PRECOMPILED=0 pip install -e . --verbose --no-build-isolation
# VLLM_USE_PRECOMPILED=0 - Force compilation of ALL CUDA kernels from source, build all kernels, not needed if only modifying Python code
```



