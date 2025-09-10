### 1. Quick‑look: Open the Nsight Systems result
1. **Locate the “.nsys-rep” file** that was created by: 
```bash
 nsys profile vllm serve
  ``` 
By default it’s saved to the current working directory.
2. **Open it in Nsight Systems GUI** (or Nsight Systems 34+ CLI). 
- In the GUI, you’ll instantly see the **Timeline**, **GPU Utilization**, **API Call Graph**, **CPU** and **GPU Threads/Processes** panels. 
- If you’re on a headless server, use the CLI export:
```bash
 nsys stats --display=NVIDIA_GPU --delimiter=, vllm.nsys-rep > stats.csv
 ```
 3. **Zoom into the chat session window.** 
 The profile shows the *full serve* lifecycle: model load, token generation loop, and any waiting.
 ---
 ### 2. Identify the main bottlenecks
| Area | What to look for | Why it matters | Typical signs |
|------|-----------------|----------------|---------------|
| **CPU–GPU Sync** | CPU waiting for GPU (or vice versa). | Consumes wall‑time without work. | Large gaps on the timeline where GPU is idle while CPU is busy or vice‑versa. |
| **Kernel Time** | Aggregated time spent in kernels. | Indicates compute‑bound part of the pipeline. | If a single kernel dominates >70% of overall compute time. |
| **Memory Bandwidth / Transfers** | Data moved between CPU & GPU. | Bottlenecks for large batch size or KV caches. | High bytes transferred but low throughput. |
| **API Latency** | Calls to cuBLAS, cuDNN, cuSparse, RAFT, etc. | Overhead from frequent calls or sub‑optimal launch configurations. | Spike in “API latency” even if kernel time is short. |
| **Batch Job Queue** | Queue of dispatched jobs vs completed jobs. | For vLLM, the async scheduler may wait on multiple requests. | Long waiting queue before the CUDA kernel launch. |

#### Quick test: Are you *bus‑bound* or *compute‑bound*?
```bash
sys graph -f json vllm.nsys-rep | tee graph.json
```
Open `graph.json` in a viewer and check the **GPU Utilization** vs **Memory Utilization** curves integrated over time.
---
### 3. Drill‑down into the *Yield* – the *token‑generation* stage
vLLM typically performs the following per token:
| Step | CUDA kernel | Empirical tuning knobs |
|------|-------------|-----------------------|
| **Embedding** | `sgemm` (or `dscalc`) | `--batch_size`, `max_batch_size` |
| **Self‑Attention** | `bmm` + `softmax` + `elementwise` | `--rope_scaling`, `--flash_attention` |
| **MLP** | `bmm` + `gelu` | `--tensor_parallelism`, `--pipeline_parallelism` |
| **Bos/Positional embeddings** **Cache** | custom cache ops | `--kv_cache_dtype` (fp16 vs int8) |
**How to spot the problematic step?**
**Timeline** – Highlight each of the GPU API calls, color‑code “Request Generation”, “Attention”, “MLP”.
**Kernel Metrics** – For each kernel, look at:
 - `GPU Seconds / Instance`
 - `Instructions per Cycle (IPC)`
 - `0% Branch Divergence`
 - `Warp Efficiency` 

**Memory flow** 
 – In the **CPU ↔ GPU** transfer panel, check transfer times for `kv_cache` and `output` buffers.

---

### 4. What to change next?

| Potential fix | Why it helps | Implementation |
|---------------|--------------|----------------|
| **Reduce Client‑side stalls** | CPU can wait for GPU; the server can prepare next request. | Increase `max_seq_len` or request pre‑fetching in vLLM config; bump `num_concurrent` to yield more dynamic load. |
| **Launch **12‑bit float** (FP16/INT8)** | Uses fewer bytes per element → higher memory bandwidth & larger batch per GPU. | `--dtype=float16` or `--dtype=bfloat16`; convert model to quantized INT8 if possible. |
| **Use flash‑attention – Enable** | Pregel reduced memory alphabet (q, k, v). | `--flash-attn` flag, ensure CUDA 12.1+, cuBLAS 12+. |
| **Adjust `max_batch_size`** | A proper batch will keep the GPU busy across token steps. | Experiment between 8–32 for 40 GB GPUs; find sweet spot where GPU occupancy ~90% but GPU stalls minimal. |
| **Profile with Nsight Compute** | Kernel‑level view of missed occupancy, memory traffic. | `ncu -o ncu.traces --set full vllm serve` – inspect `L1 Read/Write` and `Memory Utilization`. |
| **Change the scheduler** | vLLM uses a “sequence‑aware” scheduler; tuning `max_wait_time` and queue length can reduce waiting. | `--scheduler=bh` or `async` – try `--max_wait_time=1ms`. |

> **Tip**: The first *few* token generations are usually far more expensive because the *KV cache* is empty (no beam reuse). Profiling after caching the first few tokens often shows a different pattern.

---

### 5. Validate your changes

1. **Re‑run Nsight Systems for a short burst** (10–20 requests). 
2. **Save and compare** the key metrics:
 - `GPU Utilization %`
 - `Sustained throughput` (tokens/sec per GPU)
 - `Latency per token` (mean, 95th percentile)
3. If you see a **>10–20% drop in utilisation** or a **>2× latency decrease**, the tweak helped.
> **Side‑check**: Compare CPU utilisation before/after. A significant drop indicates that you moved more work onto the GPU.
---

### 6. Keep iterating, but also remember the big picture

* **Incrementally tweak configuration knobs** – don’t change several at once. 
*  **Document** the change, baseline measurement, and result. 
* **Monitor** for side‑effects: memory consumption growth, out‑of‑memory crashes, or degraded warm‑up times.

---

### 7. Advanced: Use HuggingFace - Accelerator API Traces

If you need deeper insight into how HuggingFace calls map to the kernels, enable trace logging:

```bash
HF_TRACE=1 vllm serve --model OpenAI/gpt-4o
```

This prints a high‑level trace of every token generation call that can be juxtaposed with the GPU timeline.

---

## TL;DR Checklist
1. **Open** the `.nsys-rep` → **Timeline** + **API Call Graph**. 
2. **Spot** big gaps → CPU–GPU sync, heavy memory transfer, single kernel dominator. 
3. **Tune** one knob at a time: batch size, dtype, flash‑attention, KV cache format. 
4. **Re‑profile** → compare `GPU %`, `Tokens/s`, `Latency`. 
5. **Document** every change → baseline & result. \n\nGood luck, and enjoy squeezing every ounce of performance out of your vLLM deployment!
