#!/usr/bin/env python3
"""
Anomaly Detection System - Main Entry Point

This script provides a unified interface for the anomaly detection system:
- Collect normal data
- Collect anomaly data  
- Train models
- Run real-time detection

Usage:
    python main.py detect           # Run real-time detection
    python main.py collect-normal   # Collect normal behavior data
    python main.py collect-anomaly  # Collect anomaly data with CPU stress
    python main.py train            # Train and select models
    python main.py demo             # Run a quick demo
"""

import argparse
import sys
import time
from pathlib import Path

from config import ensure_directories


def cmd_detect(args):
    """Run real-time anomaly detection."""
    from detector.runtime_detector import RuntimeDetector
    
    detector = RuntimeDetector(sampling_interval=args.interval)
    
    try:
        detector.run(
            duration_seconds=args.duration,
            log_to_file=not args.no_log,
            verbose=not args.quiet,
        )
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease train the model first:")
        print("  python main.py train")
        sys.exit(1)


def cmd_collect_normal(args):
    """Collect normal behavior data."""
    from collect_normal_data import collect_normal_data
    
    collect_normal_data(
        duration_minutes=args.duration,
        sampling_interval=args.interval,
    )


def cmd_collect_anomaly(args):
    """Collect anomaly data with CPU stress."""
    from collect_anomaly_data import collect_anomaly_data, collect_intermittent_stress
    
    if args.intermittent:
        collect_intermittent_stress(
            total_duration_minutes=args.duration,
            stress_interval_seconds=args.stress_interval,
            stress_duration_seconds=args.stress_duration,
            num_workers=args.workers,
        )
    else:
        collect_anomaly_data(
            stress_duration_seconds=args.duration * 60,
            num_workers=args.workers,
        )


def cmd_train(args):
    """Train and select anomaly detection models."""
    from train import run_training_pipeline
    
    run_training_pipeline(
        skip_data_prep=args.skip_data_prep,
        tune_hyperparams=not args.no_tuning,
        include_unsupervised=not args.supervised_only,
    )


def cmd_full_pipeline(args):
    """Run the complete pipeline: collect data, train, and detect."""
    print("=" * 70)
    print("FULL PIPELINE EXECUTION")
    print("=" * 70)
    
    # Step 1: Collect normal data
    print("\n[Step 1/4] Collecting normal data (2 minutes)...")
    print("Please use your laptop normally during this phase.")
    print("-" * 70)
    
    from collect_normal_data import collect_normal_data
    collect_normal_data(duration_minutes=2)
    
    # Step 2: Collect anomaly data
    print("\n[Step 2/4] Collecting anomaly data with CPU stress...")
    print("-" * 70)
    
    from collect_anomaly_data import collect_anomaly_data
    collect_anomaly_data(stress_duration_seconds=60, warmup_seconds=10, cooldown_seconds=10)
    
    # Step 3: Train models
    print("\n[Step 3/4] Training anomaly detection models...")
    print("-" * 70)
    
    from train import run_training_pipeline
    run_training_pipeline(tune_hyperparams=False)  # Fast training for demo
    
    # Step 4: Run detection
    print("\n[Step 4/4] Running real-time detection (30 seconds)...")
    print("-" * 70)
    
    from detector.runtime_detector import RuntimeDetector
    detector = RuntimeDetector()
    detector.run(duration_seconds=30)
    
    print("\n" + "=" * 70)
    print("FULL PIPELINE COMPLETE!")
    print("=" * 70)


def main():
    ensure_directories()
    
    parser = argparse.ArgumentParser(
        description="Anomaly Detection System for Laptop/Workstation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py detect                  # Run real-time detection
    python main.py detect -d 60            # Detect for 60 seconds
    python main.py collect-normal -d 10    # Collect 10 minutes of normal data
    python main.py collect-anomaly -d 2    # Collect 2 minutes of anomaly data
    python main.py train                   # Train models
    python main.py full-pipeline           # Run complete pipeline
    
    # To simulate attacks during detection, use stress.py in another terminal:
    python stress.py                       # Run CPU stress attack
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Detect command
    detect_parser = subparsers.add_parser(
        "detect",
        help="Run real-time anomaly detection"
    )
    detect_parser.add_argument(
        "-d", "--duration",
        type=float,
        default=None,
        help="Detection duration in seconds (default: run until Ctrl+C)",
    )
    detect_parser.add_argument(
        "-i", "--interval",
        type=float,
        default=1.0,
        help="Sampling interval in seconds (default: 1.0)",
    )
    detect_parser.add_argument(
        "--no-log",
        action="store_true",
        help="Don't log to file",
    )
    detect_parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet mode",
    )
    
    # Collect normal command
    normal_parser = subparsers.add_parser(
        "collect-normal",
        help="Collect normal behavior data"
    )
    normal_parser.add_argument(
        "-d", "--duration",
        type=float,
        default=10,
        help="Duration in minutes (default: 10)",
    )
    normal_parser.add_argument(
        "-i", "--interval",
        type=float,
        default=1.0,
        help="Sampling interval in seconds (default: 1.0)",
    )
    
    # Collect anomaly command
    anomaly_parser = subparsers.add_parser(
        "collect-anomaly",
        help="Collect anomaly data with CPU stress"
    )
    anomaly_parser.add_argument(
        "-d", "--duration",
        type=float,
        default=2,
        help="Duration in minutes (default: 2)",
    )
    anomaly_parser.add_argument(
        "-w", "--workers",
        type=int,
        default=12,
        help="Number of stress worker processes (default: 12 for your CPU)",
    )
    anomaly_parser.add_argument(
        "--intermittent",
        action="store_true",
        help="Use intermittent stress pattern",
    )
    anomaly_parser.add_argument(
        "--stress-interval",
        type=float,
        default=30,
        help="Seconds between stress periods (default: 30)",
    )
    anomaly_parser.add_argument(
        "--stress-duration",
        type=float,
        default=15,
        help="Duration of each stress period (default: 15)",
    )
    
    # Train command
    train_parser = subparsers.add_parser(
        "train",
        help="Train anomaly detection models"
    )
    train_parser.add_argument(
        "--skip-data-prep",
        action="store_true",
        help="Skip data preparation",
    )
    train_parser.add_argument(
        "--no-tuning",
        action="store_true",
        help="Skip hyperparameter tuning",
    )
    train_parser.add_argument(
        "--supervised-only",
        action="store_true",
        help="Train only supervised models",
    )
    
    # Full pipeline command
    pipeline_parser = subparsers.add_parser(
        "full-pipeline",
        help="Run complete pipeline (collect, train, detect)"
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    # Dispatch to command handler
    commands = {
        "detect": cmd_detect,
        "collect-normal": cmd_collect_normal,
        "collect-anomaly": cmd_collect_anomaly,
        "train": cmd_train,
        "full-pipeline": cmd_full_pipeline,
    }
    
    cmd_handler = commands.get(args.command)
    if cmd_handler:
        cmd_handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
