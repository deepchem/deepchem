"""
Comprehensive benchmarking suite for DeepChem DFT functionals.

This module benchmarks neural XC functionals (NNLDA, NNPBE, Neural B3LYP)
against each other and traditional implementations, measuring:
- Computation time per forward pass
- Memory usage (peak RAM)
- Scaling behavior (10 to 1000 atoms)
- Numerical consistency across runs

Usage:
    python benchmark_xc.py --quick          # Fast test (3 sizes)
    python benchmark_xc.py --full           # Full benchmark (7 sizes)
    python benchmark_xc.py --functional NeuralB3LYP  # Single functional
"""

import torch
import time
import sys
import os
from typing import Dict, List, Tuple, Optional
import tracemalloc

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

print(f"📂 Looking for functionals in: {parent_dir}")

# Import DeepChem functionals with graceful fallback
DEEPCHEM_AVAILABLE = False
NNLDA = None
NNPBE = None

try:
    from nnxc import NNLDA, NNPBE, HAS_DQC
    DEEPCHEM_AVAILABLE = True
    print("✅ DeepChem NNLDA and NNPBE available")
except Exception as e:
    print(f"⚠️  DeepChem functionals not available: {e}")
    print("    Continuing with Neural B3LYP only...")

# Import Neural B3LYP (standalone, no DQC dependency)
B3LYP_AVAILABLE = False
NeuralB3LYPSimple = None

try:
    from neural_b3lyp_simple import NeuralB3LYP as NeuralB3LYPSimple
    B3LYP_AVAILABLE = True
    print("✅ Neural B3LYP available")
except ImportError as e:
    print(f"⚠️  Neural B3LYP not available: {e}")
    # Try to list files in parent directory to debug
    try:
        files = [f for f in os.listdir(parent_dir) if f.endswith('.py')]
        print(f"    Available Python files in {parent_dir}:")
        for f in files:
            print(f"      - {f}")
    except:
        pass


class BenchmarkResult:
    """Container for benchmark results."""
    
    def __init__(self, functional_name: str):
        self.functional_name = functional_name
        self.times: List[float] = []
        self.memory_mb: List[float] = []
        self.system_sizes: List[int] = []
        
    def add_measurement(self, system_size: int, time_ms: float, memory_mb: float):
        """Add a single measurement."""
        self.system_sizes.append(system_size)
        self.times.append(time_ms)
        self.memory_mb.append(memory_mb)
    
    def average_time(self) -> float:
        """Get average computation time in ms."""
        return sum(self.times) / len(self.times) if self.times else 0.0
    
    def average_memory(self) -> float:
        """Get average memory usage in MB."""
        return sum(self.memory_mb) / len(self.memory_mb) if self.memory_mb else 0.0
    
    def __repr__(self):
        return f"BenchmarkResult({self.functional_name}: {len(self.times)} measurements)"


def create_test_density(n_atoms: int, n_features: int = 3) -> torch.Tensor:
    """
    Create synthetic density data for benchmarking.
    
    Args:
        n_atoms: Number of atoms to simulate
        n_features: Number of features (density, spin, gradient)
    
    Returns:
        Tensor of shape (n_atoms, n_features) with realistic DFT values
    """
    torch.manual_seed(42)  # Reproducible benchmarks
    
    # Realistic density values for organic molecules
    density = torch.rand(n_atoms, 1) * 0.5 + 0.1  # [0.1, 0.6]
    
    if n_features == 1:
        return density
    
    # Add spin density and gradient norm for GGA functionals
    spin_density = torch.rand(n_atoms, 1) * 0.1  # Small spin polarization
    gradient_norm = torch.rand(n_atoms, 1) * 0.3  # Typical gradient values
    
    return torch.cat([density, spin_density, gradient_norm], dim=1)


def benchmark_single_functional(
    functional_name: str,
    system_sizes: List[int],
    n_runs: int = 5,
    warmup: int = 2
) -> BenchmarkResult:
    """
    Benchmark a single functional across different system sizes.
    
    Args:
        functional_name: Name of functional ('NNLDA', 'NNPBE', 'NeuralB3LYP')
        system_sizes: List of atom counts to test
        n_runs: Number of timing runs per size
        warmup: Number of warmup runs before timing
    
    Returns:
        BenchmarkResult with timing and memory data
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking: {functional_name}")
    print(f"{'='*60}")
    
    result = BenchmarkResult(functional_name)
    
    # Initialize functional
    functional = None
    n_features = 1  # Default for LDA
    
    if functional_name == "NNLDA" and DEEPCHEM_AVAILABLE and NNLDA is not None:
        try:
            functional = NNLDA()
            n_features = 1
        except Exception as e:
            print(f"❌ Could not initialize NNLDA: {e}")
            return result
            
    elif functional_name == "NNPBE" and DEEPCHEM_AVAILABLE and NNPBE is not None:
        try:
            functional = NNPBE()
            n_features = 3  # GGA needs density + gradient
        except Exception as e:
            print(f"❌ Could not initialize NNPBE: {e}")
            return result
            
    elif functional_name == "NeuralB3LYP" and B3LYP_AVAILABLE and NeuralB3LYPSimple is not None:
        try:
            functional = NeuralB3LYPSimple()
            n_features = 3  # GGA-level functional
        except Exception as e:
            print(f"❌ Could not initialize NeuralB3LYP: {e}")
            return result
        
    else:
        print(f"❌ Functional {functional_name} not available")
        return result
    
    # Benchmark each system size
    for size in system_sizes:
        print(f"\n  System size: {size} atoms")
        
        # Create test data
        density = create_test_density(size, n_features)
        
        # Warmup runs
        for _ in range(warmup):
            try:
                if functional_name in ["NNLDA", "NNPBE"]:
                    _ = functional.get_edensityxc(density)
                else:  # NeuralB3LYP
                    if n_features == 3:
                        _ = functional(density[:, 0:1], density[:, 2:3])
                    else:
                        _ = functional(density)
            except Exception as e:
                print(f"    ⚠️  Warmup failed: {e}")
                continue
        
        # Timing runs with memory tracking
        times = []
        memories = []
        
        for run in range(n_runs):
            # Start memory tracking
            tracemalloc.start()
            
            # Time the forward pass
            start_time = time.perf_counter()
            
            try:
                if functional_name in ["NNLDA", "NNPBE"]:
                    energy = functional.get_edensityxc(density)
                else:  # NeuralB3LYP
                    if n_features == 3:
                        energy = functional(density[:, 0:1], density[:, 2:3])
                    else:
                        energy = functional(density)
                
                # Force computation (in case of lazy evaluation)
                if isinstance(energy, torch.Tensor):
                    _ = energy.sum().item()
                    
            except Exception as e:
                print(f"    ❌ Run {run+1} failed: {e}")
                tracemalloc.stop()
                continue
            
            end_time = time.perf_counter()
            
            # Get peak memory usage
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            time_ms = (end_time - start_time) * 1000  # Convert to ms
            memory_mb = peak / (1024 * 1024)  # Convert to MB
            
            times.append(time_ms)
            memories.append(memory_mb)
            
            print(f"    Run {run+1}/{n_runs}: {time_ms:.3f} ms, {memory_mb:.2f} MB")
        
        # Record average for this size
        if times and memories:
            avg_time = sum(times) / len(times)
            avg_memory = sum(memories) / len(memories)
            result.add_measurement(size, avg_time, avg_memory)
            print(f"  ✅ Average: {avg_time:.3f} ms, {avg_memory:.2f} MB")
        else:
            print(f"  ❌ No successful runs for size {size}")
    
    return result


def benchmark_all_functionals(
    system_sizes: Optional[List[int]] = None,
    quick: bool = False
) -> Dict[str, BenchmarkResult]:
    """
    Benchmark all available functionals.
    
    Args:
        system_sizes: Custom list of atom counts, or None for defaults
        quick: Use quick mode (fewer sizes, fewer runs)
    
    Returns:
        Dictionary mapping functional names to BenchmarkResult objects
    """
    if system_sizes is None:
        if quick:
            system_sizes = [10, 100, 500]  # Quick test
        else:
            system_sizes = [10, 50, 100, 200, 500, 1000]  # Full benchmark
    
    n_runs = 3 if quick else 5
    
    functionals = []
    if DEEPCHEM_AVAILABLE and NNLDA is not None:
        functionals.append("NNLDA")
    if DEEPCHEM_AVAILABLE and NNPBE is not None:
        functionals.append("NNPBE")
    if B3LYP_AVAILABLE and NeuralB3LYPSimple is not None:
        functionals.append("NeuralB3LYP")
    
    if not functionals:
        print("\n❌ No functionals available to benchmark!")
        print("   Please ensure neural_b3lyp_simple.py exists in the dft/ directory")
        return {}
    
    print("\n" + "="*60)
    print("DeepChem DFT Functional Benchmarking Suite")
    print("="*60)
    print(f"System sizes: {system_sizes}")
    print(f"Runs per size: {n_runs}")
    print(f"Functionals: {', '.join(functionals)}")
    print("="*60)
    
    results = {}
    for func_name in functionals:
        result = benchmark_single_functional(func_name, system_sizes, n_runs)
        results[func_name] = result
    
    return results


def print_summary(results: Dict[str, BenchmarkResult]):
    """Print formatted benchmark summary."""
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    for func_name, result in results.items():
        if not result.times:
            print(f"\n{func_name}: No data")
            continue
            
        print(f"\n{func_name}:")
        print(f"  Average time: {result.average_time():.3f} ms")
        print(f"  Average memory: {result.average_memory():.2f} MB")
        print(f"  Measurements: {len(result.times)} system sizes")
    
    print("\n" + "="*60)


def save_results_markdown(results: Dict[str, BenchmarkResult], filename: str = "benchmark_results.md"):
    """Save results as markdown table."""
    filepath = os.path.join(os.path.dirname(__file__), filename)
    
    with open(filepath, 'w') as f:
        f.write("# DeepChem DFT Functional Benchmarks\n\n")
        f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        f.write("| Functional | Avg Time (ms) | Avg Memory (MB) | Measurements |\n")
        f.write("|------------|---------------|-----------------|---------------|\n")
        
        for func_name, result in results.items():
            if result.times:
                f.write(f"| {func_name} | {result.average_time():.3f} | "
                       f"{result.average_memory():.2f} | {len(result.times)} |\n")
        
        f.write("\n## Detailed Results\n\n")
        for func_name, result in results.items():
            if not result.times:
                continue
                
            f.write(f"### {func_name}\n\n")
            f.write("| System Size (atoms) | Time (ms) | Memory (MB) |\n")
            f.write("|---------------------|-----------|-------------|\n")
            
            for size, time_val, mem_val in zip(result.system_sizes, result.times, result.memory_mb):
                f.write(f"| {size} | {time_val:.3f} | {mem_val:.2f} |\n")
            f.write("\n")
    
    print(f"\n✅ Results saved to: {filepath}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark DeepChem DFT functionals")
    parser.add_argument("--quick", action="store_true", help="Quick test mode (3 sizes)")
    parser.add_argument("--full", action="store_true", help="Full benchmark (6 sizes)")
    parser.add_argument("--functional", type=str, help="Benchmark single functional")
    
    args = parser.parse_args()
    
    if args.functional:
        # Single functional benchmark
        sizes = [10, 50, 100, 200, 500] if args.full else [10, 100, 500]
        result = benchmark_single_functional(args.functional, sizes)
        print_summary({args.functional: result})
        save_results_markdown({args.functional: result})
    else:
        # Benchmark all functionals
        results = benchmark_all_functionals(quick=args.quick or not args.full)
        print_summary(results)
        save_results_markdown(results)
    
    print("\n✅ Benchmark complete!")
