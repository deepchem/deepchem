"""
Visualization tools for DFT benchmark results.

Creates publication-quality charts comparing functional performance.
Requires matplotlib (install with: pip install matplotlib)
"""

import sys
import os

# Add parent for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib not available - install with: pip install matplotlib")

from benchmark_xc import benchmark_all_functionals, BenchmarkResult
from typing import Dict


def plot_time_scaling(results: Dict[str, BenchmarkResult], output_file: str = "time_scaling.png"):
    """
    Plot computation time vs system size for all functionals.
    
    Args:
        results: Dictionary of benchmark results
        output_file: Output filename for the plot
    """
    if not MATPLOTLIB_AVAILABLE:
        print("❌ Cannot create plots without matplotlib")
        return
    
    plt.figure(figsize=(10, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']
    
    for idx, (func_name, result) in enumerate(results.items()):
        if not result.times:
            continue
        plt.plot(result.system_sizes, result.times, 
                marker=markers[idx % len(markers)],
                color=colors[idx % len(colors)],
                linewidth=2, markersize=8, label=func_name)
    
    plt.xlabel('System Size (number of atoms)', fontsize=12)
    plt.ylabel('Computation Time (ms)', fontsize=12)
    plt.title('DFT Functional Performance Scaling', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filepath = os.path.join(os.path.dirname(__file__), output_file)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✅ Time scaling plot saved: {filepath}")
    plt.close()


def plot_memory_scaling(results: Dict[str, BenchmarkResult], output_file: str = "memory_scaling.png"):
    """Plot memory usage vs system size."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    plt.figure(figsize=(10, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['s', 'o', '^', 'D', 'v']
    
    has_data = False
    for idx, (func_name, result) in enumerate(results.items()):
        if not result.memory_mb or all(m == 0.0 for m in result.memory_mb):
            continue
        has_data = True
        plt.plot(result.system_sizes, result.memory_mb, 
                marker=markers[idx % len(markers)],
                color=colors[idx % len(colors)],
                linewidth=2, markersize=8, label=func_name)
    
    if not has_data:
        print("⚠️  No memory data to plot (all values are 0.00 MB)")
        plt.close()
        return
    
    plt.xlabel('System Size (number of atoms)', fontsize=12)
    plt.ylabel('Memory Usage (MB)', fontsize=12)
    plt.title('DFT Functional Memory Consumption', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filepath = os.path.join(os.path.dirname(__file__), output_file)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✅ Memory scaling plot saved: {filepath}")
    plt.close()


def plot_comparison_bar(results: Dict[str, BenchmarkResult], output_file: str = "comparison.png"):
    """Create bar chart comparing average performance."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    func_names = [name for name in results.keys() if results[name].times]
    if not func_names:
        print("⚠️  No data to plot")
        return
    
    avg_times = [results[name].average_time() for name in func_names]
    avg_mems = [results[name].average_memory() for name in func_names]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Time comparison
    bars1 = ax1.bar(func_names, avg_times, color=colors[:len(func_names)])
    ax1.set_ylabel('Average Time (ms)', fontsize=12)
    ax1.set_title('Average Computation Time', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10)
    
    # Memory comparison
    if any(m > 0.01 for m in avg_mems):
        bars2 = ax2.bar(func_names, avg_mems, color=colors[:len(func_names)])
        ax2.set_ylabel('Average Memory (MB)', fontsize=12)
        ax2.set_title('Average Memory Usage', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=10)
    else:
        ax2.text(0.5, 0.5, 'Memory data not available\n(all values < 0.01 MB)',
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Average Memory Usage', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    filepath = os.path.join(os.path.dirname(__file__), output_file)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✅ Comparison bar chart saved: {filepath}")
    plt.close()


def create_all_visualizations(quick: bool = False):
    """Run benchmarks and create all visualization charts."""
    print("🚀 Running benchmarks...")
    results = benchmark_all_functionals(quick=quick)
    
    if not results or not any(r.times for r in results.values()):
        print("❌ No benchmark data to visualize")
        return
    
    print("\n📊 Creating visualizations...")
    plot_time_scaling(results)
    plot_memory_scaling(results)
    plot_comparison_bar(results)
    
    print("\n✅ All visualizations complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize DFT benchmark results")
    parser.add_argument("--quick", action="store_true", help="Quick benchmark mode")
    
    args = parser.parse_args()
    
    if not MATPLOTLIB_AVAILABLE:
        print("❌ Please install matplotlib: pip install matplotlib")
        sys.exit(1)
    
    create_all_visualizations(quick=args.quick)
