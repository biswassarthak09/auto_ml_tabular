"""
CUDA Diagnostic Script for AutoML Pipeline
Run this on your NVIDIA machine to diagnose GPU detection issues
"""

import sys
import os

print("🔍 CUDA Diagnostic Report")
print("=" * 50)

# 1. Check PyTorch installation and CUDA availability
print("\n1. PyTorch Installation Check:")
try:
    import torch
    print(f"✅ PyTorch version: {torch._version_}")
    print(f"✅ PyTorch CUDA compiled: {torch.version.cuda}")
    print(f"✅ PyTorch CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"✅ Current CUDA device: {torch.cuda.current_device()}")
        print(f"✅ CUDA capability: {torch.cuda.get_device_capability()}")
    else:
        print("❌ CUDA not available in PyTorch")
except ImportError:
    print("❌ PyTorch not installed")
except Exception as e:
    print(f"❌ PyTorch error: {e}")

# 2. Check NVIDIA drivers and CUDA runtime
print("\n2. System CUDA Check:")
try:
    import subprocess
    
    # Check nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ nvidia-smi available")
            lines = result.stdout.split('\n')
            for line in lines:
                if 'CUDA Version' in line:
                    print(f"✅ {line.strip()}")
                    break
        else:
            print("❌ nvidia-smi failed")
    except FileNotFoundError:
        print("❌ nvidia-smi not found")
    except Exception as e:
        print(f"❌ nvidia-smi error: {e}")
        
    # Check nvcc
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'release' in line.lower():
                    print(f"✅ NVCC: {line.strip()}")
                    break
        else:
            print("❌ nvcc not available")
    except FileNotFoundError:
        print("❌ nvcc not found")
    except Exception as e:
        print(f"❌ nvcc error: {e}")
        
except Exception as e:
    print(f"❌ System check error: {e}")

# 3. Test device creation
print("\n3. Device Creation Test:")
try:
    import torch
    
    # Test auto detection
    auto_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Auto device: {auto_device}")
    
    # Test explicit CUDA
    if torch.cuda.is_available():
        cuda_device = torch.device('cuda')
        print(f"✅ Explicit CUDA device: {cuda_device}")
        
        # Test tensor operations
        test_tensor = torch.randn(10, 10).to(cuda_device)
        print(f"✅ CUDA tensor created: {test_tensor.device}")
        
        # Test computation
        result = torch.mm(test_tensor, test_tensor.t())
        print(f"✅ CUDA computation successful: {result.device}")
    else:
        print("❌ Cannot test CUDA - not available")
        
except Exception as e:
    print(f"❌ Device test error: {e}")

# 4. Test AutoML components
print("\n4. AutoML Components CUDA Test:")
try:
    # Test Meta-Learning
    sys.path.append('/Users/sarthakbiswas/Documents/automl/auto_ml_tabular')
    from src.automl_pipeline.meta_learning import AdvancedMetaLearningAutoML
    
    meta_model = AdvancedMetaLearningAutoML('test_cuda_check')
    print(f"✅ Meta-learning device: {meta_model.device}")
    
    # Test NAS-HPO
    from src.automl_pipeline.nas_hpo_optuna import NASHPOOptimizer, PyTorchMLP
    
    nas_optimizer = NASHPOOptimizer('data', 'test_cuda_check')
    print(f"✅ NAS-HPO CUDA available: {nas_optimizer.cuda_available}")
    
    # Test PyTorch MLP
    mlp = PyTorchMLP(device='auto')
    print(f"✅ PyTorch MLP device: {mlp.device}")
    
except Exception as e:
    print(f"❌ AutoML components error: {e}")
    import traceback
    traceback.print_exc()

# 5. Environment variables
print("\n5. Environment Variables:")
cuda_vars = ['CUDA_HOME', 'CUDA_PATH', 'CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES']
for var in cuda_vars:
    value = os.environ.get(var, 'Not set')
    print(f"   {var}: {value}")

print("\n" + "=" * 50)
print("🎯 Diagnosis Complete!")
print("\n💡 If CUDA is available but AutoML uses CPU:")
print("   1. Check if PyTorch was installed with CUDA support")
print("   2. Verify CUDA versions match (PyTorch CUDA vs System CUDA)")
print("   3. Check CUDA_VISIBLE_DEVICES environment variable")
print("   4. Try: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")