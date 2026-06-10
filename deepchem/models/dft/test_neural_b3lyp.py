"""
Test Neural B3LYP - Both Simple and Production Versions
"""
import sys
import torch

print("=" * 70)
print("TESTING NEURAL B3LYP - SIMPLE vs PRODUCTION")
print("=" * 70)

# Test 1: Simple version (always works)
print("\n### TEST 1: Simple Version (Learning Implementation) ###")
try:
    from neural_b3lyp_simple import NeuralB3LYP as SimpleB3LYP
    
    simple_xc = SimpleB3LYP()
    density = torch.linspace(0.1, 1.0, 5).requires_grad_()
    
    e_simple = simple_xc(density)
    print(f"✅ Simple Neural B3LYP works!")
    print(f"   Energy values: {e_simple.detach().numpy()}")
    
except Exception as e:
    print(f"❌ Simple version failed: {e}")

# Test 2: Production version (may need DQC)
print("\n### TEST 2: Production Version (DeepChem Integration) ###")
try:
    from neural_b3lyp import NeuralB3LYP, create_weight_network, HAS_DQC
    
    print(f"   DQC Available: {HAS_DQC}")
    
    if HAS_DQC:
        print("   ✅ Using LibXC-based implementation")
        weight_net = create_weight_network([32, 16])
        prod_xc = NeuralB3LYP(weight_net)
        print(f"   ✅ Production Neural B3LYP created!")
        print(f"   Functional family: {prod_xc.family}")
    else:
        print("   ⚠️  DQC not available - using simple version as fallback")
        prod_xc = NeuralB3LYP()
        density = torch.linspace(0.1, 1.0, 5)
        e_prod = prod_xc(density)
        print(f"   ✅ Fallback version works!")
        print(f"   Energy values: {e_prod.detach().numpy()}")
    
except Exception as e:
    print(f"❌ Production version error: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("\n✅ You have TWO implementations:")
print("   1. neural_b3lyp_simple.py - Standalone learning version")
print("   2. neural_b3lyp.py - Production version (PR-ready)")
print("\n📝 For GSoC PR #2:")
print("   - Submit neural_b3lyp.py (production version)")
print("   - Mention neural_b3lyp_simple.py in docs as learning resource")
print("\n🎯 Next: Add tests, documentation, and benchmark against PySCF!")
print("=" * 70)
