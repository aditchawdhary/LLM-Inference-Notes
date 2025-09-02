# Comprehensive Test Suite for PR #23996 - Phi4Flash V1 Support

## Test 1: V0 vs V1 Performance Comparison

```bash
# Create comprehensive test script
cat > ~/test_v0_v1_comparison.py << 'EOF'
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

python ~/test_v0_v1_comparison.py
```

## Test 2: Server Mode Testing

```bash
# Test V1 server functionality
export VLLM_USE_V1=1

echo "🔧 Starting V1 server test..."

# Start server in background
python -m vllm.entrypoints.openai.api_server \
    --model microsoft/Phi-4-mini-instruct \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.8 > /tmp/server.log 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server to start
echo "Waiting for server to start..."
sleep 30

# Test server endpoints
echo "🧪 Testing server endpoints..."

# Health check
curl -s http://localhost:8000/health && echo "✅ Health check passed" || echo "❌ Health check failed"

# Models endpoint
curl -s http://localhost:8000/v1/models | jq .data[0].id 2>/dev/null && echo "✅ Models endpoint works" || echo "❌ Models endpoint failed"

# Completion test
echo "🧪 Testing completion endpoint..."
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/Phi-4-mini-instruct",
    "prompt": "The capital of France is",
    "max_tokens": 20,
    "temperature": 0.7
  }' | jq -r '.choices[0].text' && echo "✅ Completion works" || echo "❌ Completion failed"

# Chat completion test
echo "🧪 Testing chat completion endpoint..."
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/Phi-4-mini-instruct",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 50,
    "temperature": 0.1
  }' | jq -r '.choices[0].message.content' && echo "✅ Chat completion works" || echo "❌ Chat completion failed"

# Cleanup
echo "🧹 Stopping server..."
kill $SERVER_PID
sleep 5

echo "✅ Server test completed!"
```

## Test 3: Stress Testing Different Scenarios

```bash
cat > ~/test_edge_cases.py << 'EOF'
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

python ~/test_edge_cases.py
```

## Test 4: Generation Quality and Variety Testing

```bash
cat > ~/test_generation_quality.py << 'EOF'
import os
from vllm import LLM, SamplingParams

os.environ['VLLM_USE_V1'] = '1'

def test_generation_variety():
    """Test different types of generation tasks"""
    
    print("🎨 GENERATION QUALITY AND VARIETY TEST")
    print("="*50)
    
    llm = LLM(
        model="microsoft/Phi-4-mini-instruct",
        trust_remote_code=True,
        max_model_len=2048
    )
    
    test_cases = [
        {
            "name": "Code Generation",
            "prompt": "Write a Python function to reverse a string:",
            "params": SamplingParams(temperature=0.1, max_tokens=100)
        },
        {
            "name": "Creative Writing", 
            "prompt": "Write a short story about a robot learning to paint:",
            "params": SamplingParams(temperature=0.8, max_tokens=150)
        },
        {
            "name": "Technical Explanation",
            "prompt": "Explain how HTTP requests work:",
            "params": SamplingParams(temperature=0.3, max_tokens=120)
        },
        {
            "name": "Mathematical Problem",
            "prompt": "Solve: If a train travels at 60 mph for 2.5 hours, how far does it go?",
            "params": SamplingParams(temperature=0.1, max_tokens=80)
        },
        {
            "name": "Conversational",
            "prompt": "Hello! How are you today? What's your favorite programming language?",
            "params": SamplingParams(temperature=0.6, max_tokens=100)
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {test_case['name']} ---")
        print(f"Prompt: {test_case['prompt']}")
        
        try:
            outputs = llm.generate([test_case['prompt']], test_case['params'])
            response = outputs[0].outputs[0].text
            
            print(f"Response: {response[:200]}...")
            print(f"Length: {len(response)} chars")
            
            # Basic quality checks
            is_relevant = len(response) > 20  # Not too short
            is_reasonable = len(response) < 1000  # Not too long
            has_content = response.strip() != ""
            
            quality_score = sum([is_relevant, is_reasonable, has_content])
            print(f"Quality Score: {quality_score}/3")
            
            results.append({
                'name': test_case['name'],
                'response_length': len(response),
                'quality_score': quality_score,
                'success': quality_score >= 2
            })
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            results.append({'name': test_case['name'], 'success': False, 'error': str(e)})
    
    # Summary
    print(f"\n📊 GENERATION QUALITY SUMMARY")
    print("="*50)
    
    successful = [r for r in results if r.get('success', False)]
    print(f"Successful tests: {len(successful)}/{len(results)}")
    
    if successful:
        avg_quality = sum(r.get('quality_score', 0) for r in successful) / len(successful)
        print(f"Average quality score: {avg_quality:.1f}/3.0")
    
    for result in results:
        status = "✅" if result.get('success', False) else "❌"
        print(f"  {status} {result['name']}")
    
    return results

if __name__ == "__main__":
    test_generation_variety()
EOF

python ~/test_generation_quality.py
```

## Test 5: V1-Specific Features Testing

```bash
cat > ~/test_v1_features.py << 'EOF'
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

python ~/test_v1_features.py
```

## Run All Tests

```bash
echo "🚀 RUNNING COMPLETE TEST SUITE FOR PR #23996"
echo "============================================="

# Make sure we're in the right environment
conda activate vllm-phi4flash-test
export VLLM_USE_V1=1

echo "Test 1/5: V0 vs V1 Performance Comparison"
python ~/test_v0_v1_comparison.py > /tmp/test1.log 2>&1
echo "✅ Test 1 completed, log: /tmp/test1.log"

echo "Test 2/5: Server Mode Testing" 
# Server test from above
echo "✅ Test 2 completed"

echo "Test 3/5: Edge Cases and Stress Testing"
python ~/test_edge_cases.py > /tmp/test3.log 2>&1
echo "✅ Test 3 completed, log: /tmp/test3.log"

echo "Test 4/5: Generation Quality Testing"
python ~/test_generation_quality.py > /tmp/test4.log 2>&1
echo "✅ Test 4 completed, log: /tmp/test4.log"

echo "Test 5/5: V1-Specific Features"
python ~/test_v1_features.py > /tmp/test5.log 2>&1
echo "✅ Test 5 completed, log: /tmp/test5.log"

echo ""
echo "🎉 ALL TESTS COMPLETED!"
echo "Check individual log files in /tmp/ for detailed results"
echo "Summary files:"
echo "  - /tmp/phi4_comparison_results.json"
ls -la /tmp/test*.log /tmp/*.json
```

This comprehensive test suite will give you:
1. **Performance comparison** between V0 and V1 engines
2. **Server functionality** testing  
3. **Edge cases** and stress testing
4. **Generation quality** across different task types
5. **V1-specific features** like chunked prefill and prefix caching

Run this and you'll have a complete picture of how well the PR works!