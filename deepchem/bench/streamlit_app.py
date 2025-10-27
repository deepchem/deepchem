"""
Streamlit web interface for DeepChem benchmarking.
"""
import os
from typing import List

import streamlit as st
import plotly.express as px

from deepchem.bench.core import BenchmarkRunner
from deepchem.bench.reports import BenchmarkReport
from deepchem.bench.models import ModelRegistry


def run_streamlit_app():
    """Run Streamlit web interface."""
    st.set_page_config(
        page_title="DeepChem Benchmark Dashboard",
        page_icon="📊",
        layout="wide")

    st.title("DeepChem Benchmark Dashboard 📊")
    st.write("Compare and analyze model performance across molecular datasets")

    # Sidebar config
    st.sidebar.header("Configuration")

    # Dataset selection
    available_datasets = [
        'tox21', 'bace', 'esol', 'freesolv', 'qm7', 'qm9', 'hiv', 'pcba',
        'muv', 'sider', 'toxcast'
    ]
    datasets = st.sidebar.multiselect("Select Datasets",
                                    available_datasets,
                                    default=['tox21'])

    # Model selection
    model_registry = ModelRegistry()
    available_models = list(model_registry.list_models().keys())
    models = st.sidebar.multiselect("Select Models",
                                  available_models,
                                  default=['graphconv'])

    # Other parameters
    split = st.sidebar.selectbox("Dataset Split Method",
                               ['random', 'scaffold', 'temporal'])

    device = st.sidebar.selectbox("Device", ['cpu', 'cuda'])
    n_jobs = st.sidebar.slider("Number of Jobs", 1, 8, 1)

    # Run button
    if st.sidebar.button("Run Benchmark"):
        if not datasets:
            st.error("Please select at least one dataset")
            return
        if not models:
            st.error("Please select at least one model")
            return

        with st.spinner("Running benchmarks..."):
            # Initialize and run benchmarks
            runner = BenchmarkRunner(
                datasets=datasets,
                models=models,
                split=split,
                n_jobs=n_jobs,
                device=device)

            results = runner.run()
            report = BenchmarkReport(results)

            # Display results
            st.header("Benchmark Results")

            # Summary table
            st.subheader("Summary Statistics")
            st.dataframe(report.summary_df)

            # Plots
            st.subheader("Performance Comparisons")
            col1, col2 = st.columns(2)

            metrics = [
                col for col in report.summary_df.columns
                if col not in ['Dataset', 'Model', 'Train Time (s)']
            ]

            for metric in metrics:
                with col1:
                    st.write(f"### {metric}")
                    fig = px.bar(report.summary_df,
                               x='Model',
                               y=metric,
                               color='Dataset',
                               barmode='group',
                               title=f'{metric} by Model and Dataset')
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.write("### Training Times")
                fig = px.bar(report.summary_df,
                           x='Model',
                           y='Train Time (s)',
                           color='Dataset',
                           barmode='group',
                           title='Training Time by Model and Dataset')
                st.plotly_chart(fig, use_container_width=True)

            # Export options
            st.header("Export Results")
            output_dir = "benchmark_results"
            os.makedirs(output_dir, exist_ok=True)

            if st.button("Generate Full Report"):
                with st.spinner("Generating report..."):
                    report.generate_html_report(output_dir)
                    st.success(
                        f"Report generated! Check the {output_dir} directory.")


if __name__ == "__main__":
    run_streamlit_app()