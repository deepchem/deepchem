#!/usr/bin/env python
"""
DeepChem Unified Benchmark CLI.

Usage:
    python -m deepchem.molnet.benchmark_dashboard --help
    
Or if installed:
    deepchem-benchmark --help
"""

from deepchem.molnet.benchmark_dashboard import run_benchmark_cli

if __name__ == '__main__':
    run_benchmark_cli()
