#!/usr/bin/env python3
"""
Script to compare and analyze experiments from W&B.

This script provides utilities for:
- Fetching experiment data from W&B
- Comparing loss curves across runs
- Generating comparison plots
- Exporting results for analysis
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def fetch_runs_from_wandb(
    project: str,
    entity: Optional[str] = None,
    tags: Optional[List[str]] = None,
    group: Optional[str] = None,
    state: str = "finished"
) -> List[Any]:
    """Fetch runs from W&B based on filters."""
    try:
        import wandb
    except ImportError:
        raise ImportError("wandb is required. Install with: pip install wandb")
    
    api = wandb.Api()
    
    # Build filters
    filters = {"state": state}
    if tags:
        filters["tags"] = {"$in": tags}
    if group:
        filters["group"] = group
    
    # Fetch runs
    path = f"{entity}/{project}" if entity else project
    runs = api.runs(path, filters=filters)
    
    return list(runs)


def extract_run_data(run) -> Dict[str, Any]:
    """Extract relevant data from a W&B run."""
    # Get config
    config = dict(run.config)
    
    # Get summary metrics
    summary = dict(run.summary)
    
    # Get history data for loss curves
    history = run.history(samples=10000)
    
    # Extract key metrics
    data = {
        "name": run.name,
        "id": run.id,
        "state": run.state,
        "tags": run.tags,
        "config": config,
        "summary": summary,
        "history": history,
        "created_at": run.created_at,
        "runtime": run.summary.get("_runtime", 0),
        "runtime_hours": run.summary.get("_runtime", 0) / 3600,
    }
    
    # Extract final metrics
    data["final_train_loss"] = summary.get("train/loss", None)
    data["final_val_loss"] = summary.get("val/loss", None)
    data["final_train_perplexity"] = summary.get("train/perplexity", None)
    data["final_val_perplexity"] = summary.get("val/perplexity", None)
    data["total_gradient_steps"] = summary.get("time/gradient_steps", None)
    data["total_tokens"] = summary.get("time/tokens_seen", None)
    data["avg_tokens_per_sec"] = summary.get("throughput/tokens_per_second", None)
    
    return data


def plot_loss_curves(
    runs_data: List[Dict[str, Any]],
    metric: str = "train/loss",
    x_axis: str = "gradient_steps",
    output_path: Optional[Path] = None,
    title: Optional[str] = None
):
    """Plot loss curves for multiple runs."""
    plt.figure(figsize=(12, 8))
    
    # Set style
    sns.set_style("whitegrid")
    colors = sns.color_palette("husl", len(runs_data))
    
    for i, run_data in enumerate(runs_data):
        history = run_data["history"]
        
        if metric not in history.columns:
            print(f"Warning: {metric} not found in run {run_data['name']}")
            continue
        
        # Determine x-axis
        if x_axis == "gradient_steps":
            x_data = history["time/gradient_steps"]
            x_label = "Gradient Steps"
        elif x_axis == "wallclock":
            x_data = history["time/wallclock_hours"]
            x_label = "Wallclock Time (hours)"
        elif x_axis == "tokens":
            x_data = history["time/tokens_seen"] / 1e6  # Convert to millions
            x_label = "Tokens Seen (millions)"
        else:
            x_data = history.index
            x_label = "Steps"
        
        # Plot
        plt.plot(
            x_data,
            history[metric],
            label=run_data["name"],
            color=colors[i],
            alpha=0.8,
            linewidth=2
        )
    
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(metric.replace("/", " ").title(), fontsize=12)
    plt.title(title or f"{metric} Comparison", fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def create_comparison_table(
    runs_data: List[Dict[str, Any]],
    metrics: List[str],
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """Create a comparison table of key metrics across runs."""
    rows = []
    
    for run_data in runs_data:
        row = {
            "name": run_data["name"],
            "model_size": run_data["config"].get("model_size", ""),
            "learning_rate": run_data["config"].get("learning_rate", ""),
            "batch_size": run_data["config"].get("batch_size", "") * 
                         run_data["config"].get("gradient_accumulation_steps", 1),
        }
        
        # Add requested metrics
        for metric in metrics:
            if "/" in metric:
                # It's a W&B metric
                row[metric] = run_data["summary"].get(metric, None)
            else:
                # It's a direct field
                row[metric] = run_data.get(metric, None)
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by final validation loss if available
    if "final_val_loss" in df.columns:
        df = df.sort_values("final_val_loss")
    
    if output_path:
        if output_path.suffix == ".csv":
            df.to_csv(output_path, index=False)
        elif output_path.suffix == ".html":
            df.to_html(output_path, index=False)
        else:
            df.to_json(output_path, orient="records", indent=2)
        print(f"Comparison table saved to {output_path}")
    
    return df


def export_run_configs(
    runs_data: List[Dict[str, Any]],
    output_dir: Path
):
    """Export configurations of all runs for reproducibility."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for run_data in runs_data:
        config_path = output_dir / f"{run_data['name']}_config.json"
        with open(config_path, 'w') as f:
            json.dump(run_data['config'], f, indent=2)
    
    print(f"Exported {len(runs_data)} configurations to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare and analyze experiments from W&B",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # W&B arguments
    parser.add_argument("--project", type=str, default="mini-lm",
                       help="W&B project name")
    parser.add_argument("--entity", type=str, default=None,
                       help="W&B entity (username or team)")
    parser.add_argument("--tags", type=str, nargs="+", default=None,
                       help="Filter runs by tags")
    parser.add_argument("--group", type=str, default=None,
                       help="Filter runs by group")
    parser.add_argument("--run-ids", type=str, nargs="+", default=None,
                       help="Specific run IDs to compare")
    
    # Comparison arguments
    parser.add_argument("--metrics", type=str, nargs="+",
                       default=["final_train_loss", "final_val_loss", 
                               "runtime_hours", "total_gradient_steps"],
                       help="Metrics to compare")
    parser.add_argument("--plot-metric", type=str, default="train/loss",
                       help="Metric to plot")
    parser.add_argument("--x-axis", type=str, default="gradient_steps",
                       choices=["gradient_steps", "wallclock", "tokens"],
                       help="X-axis for plots")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default="experiments/comparisons",
                       help="Directory to save outputs")
    parser.add_argument("--export-configs", action="store_true",
                       help="Export run configurations")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Fetching runs from W&B project: {args.project}")
    
    # Fetch runs
    if args.run_ids:
        # Fetch specific runs
        import wandb
        api = wandb.Api()
        runs = []
        for run_id in args.run_ids:
            path = f"{args.entity}/{args.project}/{run_id}" if args.entity else f"{args.project}/{run_id}"
            runs.append(api.run(path))
    else:
        # Fetch runs based on filters
        runs = fetch_runs_from_wandb(
            project=args.project,
            entity=args.entity,
            tags=args.tags,
            group=args.group
        )
    
    if not runs:
        print("No runs found matching the criteria")
        return 1
    
    print(f"Found {len(runs)} runs")
    
    # Extract data from runs
    print("Extracting run data...")
    runs_data = [extract_run_data(run) for run in runs]
    
    # Create comparison table
    print("\nCreating comparison table...")
    table_path = output_dir / "comparison_table.csv"
    df = create_comparison_table(runs_data, args.metrics, table_path)
    print("\nComparison Table:")
    print(df.to_string())
    
    # Plot loss curves
    print(f"\nPlotting {args.plot_metric} curves...")
    plot_path = output_dir / f"{args.plot_metric.replace('/', '_')}_comparison.png"
    plot_loss_curves(
        runs_data,
        metric=args.plot_metric,
        x_axis=args.x_axis,
        output_path=plot_path,
        title=f"{args.plot_metric} Comparison ({args.x_axis})"
    )
    
    # Plot validation curves if train metric was selected
    if "train" in args.plot_metric:
        val_metric = args.plot_metric.replace("train", "val")
        val_plot_path = output_dir / f"{val_metric.replace('/', '_')}_comparison.png"
        plot_loss_curves(
            runs_data,
            metric=val_metric,
            x_axis=args.x_axis,
            output_path=val_plot_path,
            title=f"{val_metric} Comparison ({args.x_axis})"
        )
    
    # Export configurations if requested
    if args.export_configs:
        config_dir = output_dir / "configs"
        export_run_configs(runs_data, config_dir)
    
    print(f"\nAll outputs saved to {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())