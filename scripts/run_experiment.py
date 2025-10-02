#!/usr/bin/env python3
"""
Script to run experiments with systematic configuration.

This script provides utilities for running experiments with:
- Configuration file support
- Hyperparameter sweeps
- Automatic experiment naming
- Batch execution
"""

import argparse
import subprocess
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import itertools
from datetime import datetime
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from mini_lm.tracking.config import ExperimentConfig


def load_base_config(config_path: Path) -> Dict[str, Any]:
    """Load base configuration from file."""
    if config_path.suffix in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        with open(config_path, 'r') as f:
            return json.load(f)


def generate_sweep_configs(
    base_config: Dict[str, Any],
    sweep_params: Dict[str, List[Any]]
) -> List[Dict[str, Any]]:
    """Generate configurations for hyperparameter sweep."""
    configs = []
    
    # Get all combinations of sweep parameters
    param_names = list(sweep_params.keys())
    param_values = [sweep_params[name] for name in param_names]
    
    for values in itertools.product(*param_values):
        # Create config for this combination
        config = base_config.copy()
        
        # Update with sweep values
        for name, value in zip(param_names, values):
            config[name] = value
        
        # Generate experiment name
        sweep_suffix = "_".join([f"{name}_{value}" for name, value in zip(param_names, values)])
        if 'experiment_name' in config:
            config['experiment_name'] = f"{config['experiment_name']}_{sweep_suffix}"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config['experiment_name'] = f"sweep_{sweep_suffix}_{timestamp}"
        
        configs.append(config)
    
    return configs


def run_single_experiment(
    config: Dict[str, Any],
    train_script: str = "train_enhanced.py",
    dry_run: bool = False
) -> bool:
    """Run a single experiment with given configuration."""
    # Save config to temporary file
    config_dir = Path("experiments/configs")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = config_dir / f"{config['experiment_name']}.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Build command
    cmd = [
        "python", train_script,
        "--config", str(config_path)
    ]
    
    # Add any command-line overrides
    if 'train_data' in config:
        cmd.extend(["--train-data", config['train_data']])
    if 'val_data' in config:
        cmd.extend(["--val-data", config['val_data']])
    
    print(f"\n{'='*60}")
    print(f"Running experiment: {config['experiment_name']}")
    print(f"Config: {config_path}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    if dry_run:
        print("DRY RUN - Skipping execution")
        return True
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ Experiment {config['experiment_name']} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Experiment {config['experiment_name']} failed with error: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠ Experiment {config['experiment_name']} interrupted by user")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run experiments with systematic configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--base-config", type=str, required=True,
                       help="Path to base configuration file")
    parser.add_argument("--train-data", type=str,
                       help="Path to training data (overrides config)")
    parser.add_argument("--val-data", type=str,
                       help="Path to validation data (overrides config)")
    
    # Sweep arguments
    sweep_group = parser.add_argument_group("Hyperparameter Sweep")
    sweep_group.add_argument("--sweep", action="store_true",
                            help="Enable hyperparameter sweep mode")
    sweep_group.add_argument("--sweep-param", action="append", nargs=2,
                            metavar=("PARAM", "VALUES"),
                            help="Parameter to sweep (e.g., --sweep-param learning_rate 1e-4,3e-4,1e-3)")
    
    # Execution arguments
    parser.add_argument("--train-script", type=str, default="train_enhanced.py",
                       help="Training script to use")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print commands without executing")
    parser.add_argument("--continue-on-error", action="store_true",
                       help="Continue running experiments even if one fails")
    
    args = parser.parse_args()
    
    # Load base configuration
    base_config = load_base_config(Path(args.base_config))
    
    # Override data paths if provided
    if args.train_data:
        base_config['train_data'] = args.train_data
    if args.val_data:
        base_config['val_data'] = args.val_data
    
    # Generate configurations
    if args.sweep and args.sweep_param:
        # Parse sweep parameters
        sweep_params = {}
        for param_name, values_str in args.sweep_param:
            # Try to parse as JSON first (for lists)
            try:
                values = json.loads(values_str)
                if not isinstance(values, list):
                    values = [values]
            except json.JSONDecodeError:
                # Fall back to comma-separated values
                values = []
                for v in values_str.split(','):
                    # Try to parse as number
                    try:
                        if '.' in v:
                            values.append(float(v))
                        else:
                            values.append(int(v))
                    except ValueError:
                        # Keep as string
                        values.append(v)
            
            sweep_params[param_name] = values
        
        configs = generate_sweep_configs(base_config, sweep_params)
        print(f"Generated {len(configs)} configurations for sweep")
    else:
        configs = [base_config]
    
    # Run experiments
    successful = 0
    failed = 0
    
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Preparing experiment...")
        
        success = run_single_experiment(
            config,
            train_script=args.train_script,
            dry_run=args.dry_run
        )
        
        if success:
            successful += 1
        else:
            failed += 1
            if not args.continue_on_error:
                print("\nStopping due to error (use --continue-on-error to continue)")
                break
    
    # Summary
    print(f"\n{'='*60}")
    print("EXPERIMENT RUN SUMMARY")
    print(f"{'='*60}")
    print(f"Total experiments: {len(configs)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped: {len(configs) - successful - failed}")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())