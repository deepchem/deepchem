"""
Report generation for benchmark results.
"""
from typing import Dict, Optional
import json
import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class BenchmarkReport:
    """Class for generating benchmark reports."""

    def __init__(self, results: Dict):
        """Initialize report generator.
        
        Parameters
        ----------
        results: Dict
            Results dictionary from BenchmarkRunner
        """
        self.results = results
        self.summary_df = self._create_summary_df()

    def _create_summary_df(self) -> pd.DataFrame:
        """Create summary DataFrame from results."""
        rows = []
        for dataset, model_results in self.results.items():
            for model, results in model_results.items():
                row = {
                    "Dataset": dataset,
                    "Model": model,
                    "Train Time (s)": results["train_time"],
                }
                row.update(results["scores"])
                rows.append(row)
        return pd.DataFrame(rows)

    def save_json(self, output_path: str):
        """Save results as JSON.
        
        Parameters
        ----------
        output_path: str
            Path to save JSON file
        """
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

    def save_csv(self, output_path: str):
        """Save summary as CSV.
        
        Parameters
        ----------
        output_path: str
            Path to save CSV file
        """
        self.summary_df.to_csv(output_path, index=False)

    def plot_metric_comparison(self, metric: str,
                             output_path: Optional[str] = None):
        """Create bar plot comparing models on a metric.
        
        Parameters
        ----------
        metric: str
            Name of metric to plot
        output_path: Optional[str]
            Path to save plot. If None, displays plot
        """
        fig = px.bar(self.summary_df,
                    x='Model',
                    y=metric,
                    color='Dataset',
                    barmode='group',
                    title=f'{metric} by Model and Dataset')

        if output_path:
            fig.write_html(output_path)
        else:
            fig.show()

    def plot_training_times(self, output_path: Optional[str] = None):
        """Create bar plot of training times.
        
        Parameters
        ----------
        output_path: Optional[str] 
            Path to save plot. If None, displays plot
        """
        fig = px.bar(self.summary_df,
                    x='Model',
                    y='Train Time (s)',
                    color='Dataset',
                    barmode='group',
                    title='Training Time by Model and Dataset')

        if output_path:
            fig.write_html(output_path)
        else:
            fig.show()

    def generate_html_report(self, output_dir: str):
        """Generate complete HTML report.
        
        Parameters
        ----------
        output_dir: str
            Directory to save report files
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # Save data
        self.save_json(os.path.join(output_dir, f'results_{timestamp}.json'))
        self.save_csv(os.path.join(output_dir, f'summary_{timestamp}.csv'))

        # Save plots
        metrics = [col for col in self.summary_df.columns
                  if col not in ['Dataset', 'Model', 'Train Time (s)']]

        for metric in metrics:
            self.plot_metric_comparison(
                metric,
                os.path.join(output_dir, f'{metric}_comparison_{timestamp}.html'))

        self.plot_training_times(
            os.path.join(output_dir, f'training_times_{timestamp}.html'))

        # Generate index.html
        html = f"""
        <html>
        <head>
            <title>DeepChem Benchmark Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ text-align: left; padding: 8px; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .plot-container {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>DeepChem Benchmark Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Summary Statistics</h2>
            {self.summary_df.to_html()}
            
            <h2>Metric Comparisons</h2>
        """

        for metric in metrics:
            html += f"""
            <div class="plot-container">
                <h3>{metric}</h3>
                <iframe src="{metric}_comparison_{timestamp}.html" 
                        width="100%" height="600px" frameborder="0"></iframe>
            </div>
            """

        html += f"""
            <div class="plot-container">
                <h3>Training Times</h3>
                <iframe src="training_times_{timestamp}.html" 
                        width="100%" height="600px" frameborder="0"></iframe>
            </div>
        </body>
        </html>
        """

        with open(os.path.join(output_dir, 'index.html'), 'w') as f:
            f.write(html)