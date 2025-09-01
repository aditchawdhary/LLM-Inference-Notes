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


```
 1  git clone https://github.com/vllm-project/vllm.git
    2  cd vllm
    3  git fetch origin pull/23996/head:pr-23996
    4  git checkout pr-23996
    5  pip install -e .
    6  python --version
    7  # Always activate before running tests
    8  conda activate vllm-pr-test
    9  # Verify you're in the right environment
   10  which python
   11  python --version
   12  conda create -n vllm-phi4flash-test python=3.10 -y
   13  conda activate vllm-phi4flash-test
   14  python --version
   15  git clone https://github.com/vllm-project/vllm.git
   16  cd vllm
   17  la
   18  ls
   19  pwd
   20  cd ..
   21  cd .
   22  cd..
   23  ls
   24  cd ..
   25  ls
   26  cd vllm
   27  ls
   28  git init
   29  git fetch origin pull/23996/head:pr-23996-phi4flash
   30  git checkout pr-23996-phi4flash
   31  git branch -v
   32  git log --oneline -n 5
   33  git diff HEAD~1 --name-only | grep -iE "(phi|flash)" || echo "Checking for model-related changes..."
   34  git diff HEAD~1 --name-only | grep -E "(model|config)" | head -5
   35  pip list | grep torch
   36  pip list 
   37  nvcc --version
   38  pip3 install -r requirements-build.txt
   39  pip3 install -r requirements/requirements-build.txt
   40  cd requirements/
   41  ls
   42  pip3 install -r build.txt
   43  pip list
   44  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   45  pip list
   46  VLLM_USE_PRECOMPILED=0 pip install -e . --verbose --no-build-isolation
   47  cd ..
   48  VLLM_USE_PRECOMPILED=0 pip install -e . --verbose --no-build-isolation
   49  pip install -e . --verbose  # Much faster - uses precompiled binaries
   50  VLLM_USE_PRECOMPILED=1 pip install -e . --verbose 
   51  export VLLM_USE_V1=1
   52  python3 << 'EOF'
from vllm import LLM, SamplingParams

print("Testing Phi4 with V1 engine...")

llm = LLM(
    model="microsoft/Phi-4-mini-instruct",
    trust_remote_code=True,
    max_model_len=1024
)

outputs = llm.generate(["What is AI?"], SamplingParams(max_tokens=50))
print("Result:", outputs[0].outputs[0].text)
print("✅ SUCCESS: Phi4 works with V1 engine!")
EOF

   53  # Create comprehensive test script
   54  cat > ~/test_v0_v1_comparison.py << 'EOF'
import os
import time
import json
from vllm import LLM, SamplingParams

def test_engine(engine_version="v1"):
    """Test Phi4 with specified engine version"""
    
    print(f"\n{'='*60}")
    print(f"Testing Phi4 with {engine_version.upper()} Engine")
    print(f"{'='*60}")
    
    # Set environment
    if engine_version == "v1":
        os.environ['VLLM_USE_V1'] = '1'
    else:
        os.environ.pop('VLLM_USE_V1', None)
    
    test_prompts = [
        "Write a Python function to calculate fibonacci numbers:",
        "Explain how machine learning works in simple terms:",
        "What are the main differences between Python and JavaScript?",
        "Describe the process of photosynthesis:",
        "How do neural networks learn from data?"
    ]
    
    results = {}
    
    try:
        # Model loading time
        print("Loading model...")
        start_time = time.time()
        
        llm = LLM(
            model="microsoft/Phi-4-mini-instruct",
            trust_remote_code=True,
            max_model_len=2048,
            gpu_memory_utilization=0.8
        )
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f} seconds")
        
        # Warmup
        llm.generate(["Hello"], SamplingParams(max_tokens=5))
        
        # Performance test
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=150,
            top_p=0.9
        )
        
        print(f"Generating responses to {len(test_prompts)} prompts...")
        gen_start = time.time()
        outputs = llm.generate(test_prompts, sampling_params)
        gen_time = time.time() - gen_start
        
        # Calculate metrics
        total_input_tokens = sum(len(prompt.split()) for prompt in test_prompts)
        total_output_tokens = sum(len(output.outputs[0].text.split()) for output in outputs)
        total_tokens = total_input_tokens + total_output_tokens
        
        throughput = total_tokens / gen_time if gen_time > 0 else 0
        
        results = {
            'engine': engine_version,
            'load_time': load_time,
            'generation_time': gen_time,
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'total_tokens': total_tokens,
            'tokens_per_second': throughput,
            'success': True
        }
        
        print(f"\n📊 {engine_version.upper()} Results:")
        print(f"   Load time: {load_time:.2f}s")
        print(f"   Generation time: {gen_time:.2f}s")  
        print(f"   Total tokens: {total_tokens}")
        print(f"   Throughput: {throughput:.2f} tokens/sec")
        
        # Show sample outputs
        print(f"\n📝 Sample Outputs:")
        for i, (prompt, output) in enumerate(zip(test_prompts[:2], outputs[:2])):
            print(f"\n--- Test {i+1} ---")
            print(f"Prompt: {prompt[:50]}...")
            print(f"Response: {output.outputs[0].text[:100]}...")
        
    except Exception as e:
        print(f"❌ {engine_version.upper()} test failed: {e}")
        results = {'engine': engine_version, 'error': str(e), 'success': False}
        import traceback
        traceback.print_exc()
    
    return results

def main():
    print("🚀 COMPREHENSIVE PHI4 V0 vs V1 COMPARISON")
    
    # Test both engines
    v0_results = test_engine("v0")
    time.sleep(5)  # Brief pause between tests
    v1_results = test_engine("v1")
    
    # Compare results
    if v0_results.get('success') and v1_results.get('success'):
        print(f"\n{'='*60}")
        print("📊 PERFORMANCE COMPARISON")
        print(f"{'='*60}")
        
        v0_throughput = v0_results['tokens_per_second']
        v1_throughput = v1_results['tokens_per_second']
        
        if v0_throughput > 0:
            speedup = v1_throughput / v0_throughput
            print(f"🚀 V1 Throughput Speedup: {speedup:.2f}x")
            
            if speedup > 1.5:
                print("✅ EXCELLENT: V1 shows significant improvement!")
            elif speedup > 1.1:
                print("✅ GOOD: V1 shows moderate improvement")
            elif speedup > 0.9:
                print("✅ OK: V1 performance comparable to V0")
            else:
                print("⚠️  WARNING: V1 performance lower than V0")
        
        load_ratio = v1_results['load_time'] / v0_results['load_time']
        print(f"📥 V1 Load Time Ratio: {load_ratio:.2f}x")
    
    # Save detailed results
    all_results = {'v0': v0_results, 'v1': v1_results, 'timestamp': time.time()}
    with open('/tmp/phi4_comparison_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to /tmp/phi4_comparison_results.json")
    return all_results

if __name__ == "__main__":
    main()
EOF

   55  python ~/test_v0_v1_comparison.py
   56  # Test V1 server functionality
   57  export VLLM_USE_V1=1
   58  echo "🔧 Starting V1 server test..."
   59  # Start server in background
   60  python -m vllm.entrypoints.openai.api_server     --model microsoft/Phi-4-mini-instruct     --trust-remote-code     --host 0.0.0.0     --port 8000     --max-model-len 2048     --gpu-memory-utilization 0.8 > /tmp/server.log 2>&1 &
   61  SERVER_PID=$!
   62  echo "Server PID: $SERVER_PID"
   63  # Wait for server to start
   64  echo "Waiting for server to start..."
   65  sleep 30
   66  # Test server endpoints
   67  echo "🧪 Testing server endpoints..."
   68  # Health check
   69  curl -s http://localhost:8000/health && echo "✅ Health check passed" || echo "❌ Health check failed"
   70  # Models endpoint
   71  curl -s http://localhost:8000/v1/models | jq .data[0].id 2>/dev/null && echo "✅ Models endpoint works" || echo "❌ Models endpoint failed"
   72  # Completion test
   73  echo "🧪 Testing completion endpoint..."
   74  curl -s http://localhost:8000/v1/completions   -H "Content-Type: application/json"   -d '{
    "model": "microsoft/Phi-4-mini-instruct",
    "prompt": "The capital of France is",
    "max_tokens": 20,
    "temperature": 0.7
  }' | jq -r '.choices[0].text' && echo "✅ Completion works" || echo "❌ Completion failed"
   75  # Chat completion test
   76  echo "🧪 Testing chat completion endpoint..."
   77  curl -s http://localhost:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "microsoft/Phi-4-mini-instruct",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 50,
    "temperature": 0.1
  }' | jq -r '.choices[0].message.content' && echo "✅ Chat completion works" || echo "❌ Chat completion failed"
   78  # Cleanup
   79  echo "🧹 Stopping server..."
   80  kill $SERVER_PID
   81  sleep 5
   82  echo "✅ Server test completed!"
   83  cat > ~/test_edge_cases.py << 'EOF'
import os
import time
from vllm import LLM, SamplingParams

os.environ['VLLM_USE_V1'] = '1'

def test_scenario(name, **llm_kwargs):
    """Test specific scenario with Phi4 V1"""
    print(f"\n🧪 Testing: {name}")
    print(f"Config: {llm_kwargs}")
    
    try:
        start_time = time.time()
        llm = LLM(
            model="microsoft/Phi-4-mini-instruct",
            trust_remote_code=True,
            **llm_kwargs
        )
        load_time = time.time() - start_time
        
        # Quick test
        outputs = llm.generate(["Hello world"], SamplingParams(max_tokens=10))
        
        print(f"✅ {name}: Load {load_time:.1f}s, Generated: '{outputs[0].outputs[0].text.strip()}'")
        return True
        
    except Exception as e:
        print(f"❌ {name}: Failed - {str(e)[:100]}")
        return False

def main():
    print("🔬 EDGE CASE AND STRESS TESTING")
    
    scenarios = [
        ("Small Context", {"max_model_len": 512}),
        ("Large Context", {"max_model_len": 4096}),
        ("Low GPU Memory", {"gpu_memory_utilization": 0.5}),
        ("High GPU Memory", {"gpu_memory_utilization": 0.95}),
        ("BF16 Precision", {"dtype": "bfloat16"}),
        ("FP16 Precision", {"dtype": "float16"}),
        ("Eager Mode", {"enforce_eager": True}),
    ]
    
    results = {}
    for name, config in scenarios:
        results[name] = test_scenario(name, **config)
        time.sleep(2)  # Brief pause between tests
    
    print(f"\n📊 EDGE CASE RESULTS:")
    passed = sum(results.values())
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    for name, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {name}")
    
    return results

if __name__ == "__main__":
    main()
EOF

   84  python ~/test_edge_cases.py
   85  python -c "import torch; torch.cuda.empty_cache()"
   86  python ~/test_edge_cases.py
   87  cat > ~/test_v1_features.py << 'EOF'
import os
import time
from vllm import LLM, SamplingParams

os.environ['VLLM_USE_V1'] = '1'

def test_v1_specific_features():
    """Test features that are specific to V1 engine"""
    
    print("⚡ V1-SPECIFIC FEATURES TEST")
    print("="*50)
    
    features_to_test = [
        {
            "name": "Chunked Prefill",
            "config": {
                "enable_chunked_prefill": True,
                "max_num_batched_tokens": 4096
            }
        },
        {
            "name": "Prefix Caching",
            "config": {
                "enable_prefix_caching": True
            }
        },
        {
            "name": "Combined Features",
            "config": {
                "enable_chunked_prefill": True,
                "enable_prefix_caching": True,
                "max_num_batched_tokens": 8192
            }
        }
    ]
    
    base_config = {
        "model": "microsoft/Phi-4-mini-instruct",
        "trust_remote_code": True,
        "max_model_len": 1024,
        "gpu_memory_utilization": 0.7
    }
    
    results = []
    
    for feature in features_to_test:
        print(f"\n🧪 Testing: {feature['name']}")
        
        try:
            config = {**base_config, **feature['config']}
            
            start_time = time.time()
            llm = LLM(**config)
            load_time = time.time() - start_time
            
            # Test with multiple similar prompts (good for prefix caching)
            prompts = [
                "Explain machine learning concepts: supervised learning",
                "Explain machine learning concepts: unsupervised learning", 
                "Explain machine learning concepts: reinforcement learning"
            ]
            
            gen_start = time.time()
            outputs = llm.generate(prompts, SamplingParams(max_tokens=50, temperature=0.1))
            gen_time = time.time() - gen_start
            
            total_tokens = sum(len(output.outputs[0].text.split()) for output in outputs)
            throughput = total_tokens / gen_time if gen_time > 0 else 0
            
            print(f"✅ {feature['name']}: Load {load_time:.1f}s, Gen {gen_time:.1f}s, {throughput:.1f} tok/s")
            
            results.append({
                'name': feature['name'],
                'load_time': load_time,
                'gen_time': gen_time,
                'throughput': throughput,
                'success': True
            })
            
        except Exception as e:
            print(f"❌ {feature['name']}: Failed - {str(e)[:100]}")
            results.append({
                'name': feature['name'],
                'success': False,
                'error': str(e)
            })
    
    # Summary
    print(f"\n📊 V1 FEATURES SUMMARY")
    print("="*50)
    
    successful = [r for r in results if r.get('success', False)]
    print(f"Working features: {len(successful)}/{len(results)}")
    
    if len(successful) >= 2:
        # Compare throughput
        throughputs = [r['throughput'] for r in successful if 'throughput' in r]
        if throughputs:
            best_throughput = max(throughputs)
            best_feature = next(r['name'] for r in successful if r.get('throughput') == best_throughput)
            print(f"Best performing config: {best_feature} ({best_throughput:.1f} tok/s)")
    
    for result in results:
        status = "✅" if result.get('success', False) else "❌"
        print(f"  {status} {result['name']}")
    
    return results

if __name__ == "__main__":
    test_v1_specific_features()
EOF

   88  python ~/test_v1_features.py
   89  history
```
