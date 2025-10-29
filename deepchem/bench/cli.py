"""
Command line interface for DeepChem benchmarking.
"""
import argparse
import os
import sys
from typing import List

from deepchem.bench.core import BenchmarkRunner
from deepchem.bench.reports import BenchmarkReport
from deepchem.bench.models import ModelRegistry


def parse_args(args: List[str]) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='DeepChem Unified Benchmark Dashboard')

    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['tox21', 'bace', 'esol', 'freesolv', 'qm7', 'qm9'],
        help='Datasets to benchmark')

    parser.add_argument(
        '--models',
        nargs='+',
        default=['graphconv', 'mpnn', 'gat', 'attentivefp'],
        help='Models to evaluate')

    parser.add_argument('--split',
                       default='random',
                       choices=['random', 'scaffold', 'temporal'],
                       help='Dataset splitting method')

    parser.add_argument('--metrics',
                       nargs='+',
                       help='Metrics to compute (default: auto-detect by task)')

    parser.add_argument('--n-jobs',
                       type=int,
                       default=1,
                       help='Number of parallel jobs')

    parser.add_argument('--device',
                       default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to run on')

    parser.add_argument('--output-dir',
                       default='benchmark_results',
                       help='Directory to save results')

    return parser.parse_args(args)


def main(args: List[str] = None):
    """Main entry point."""
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # Initialize benchmark runner
    runner = BenchmarkRunner(
        datasets=args.datasets,
        models=args.models,
        split=args.split,
        metrics=args.metrics,
        n_jobs=args.n_jobs,
        device=args.device)

    # Run benchmarks
    results = runner.run()

    # Generate reports
    report = BenchmarkReport(results)
    os.makedirs(args.output_dir, exist_ok=True)
    report.generate_html_report(args.output_dir)


if __name__ == "__main__":
    main()