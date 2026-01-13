#!/usr/bin/env python3
"""
Anomaly Detection System - Main Entry Point

Usage:
    python main.py detect              # Run real-time detection
    python main.py collect-normal      # Collect normal behavior data (10 min)
    python main.py collect-anomaly     # Collect anomaly data with CPU stress
    python main.py train               # Train and select models
"""

import sys
from config import ensure_directories


def cmd_detect():
    """Run real-time anomaly detection."""
    from detector.runtime_detector import RuntimeDetector
    
    detector = RuntimeDetector(sampling_interval=1.0)
    
    try:
        detector.run(duration_seconds=None, log_to_file=True, verbose=True)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease train the model first:")
        print("  python main.py train")
        sys.exit(1)


def cmd_collect_normal():
    """Collect normal behavior data."""
    from collect_normal_data import collect_normal_data
    collect_normal_data(duration_minutes=10)


def cmd_collect_anomaly():
    """Collect anomaly data with CPU stress."""
    from collect_anomaly_data import collect_anomaly_data
    collect_anomaly_data(stress_duration_seconds=60, warmup_seconds=5, cooldown_seconds=5)


def cmd_train():
    """Train and select anomaly detection models."""
    from train import run_training_pipeline
    run_training_pipeline(
        skip_data_prep=False,
        tune_hyperparams=True,
        include_unsupervised=True,
    )


def print_usage():
    """Print usage information."""
    print("""
Anomaly Detection System for Laptop/Workstation
================================================

Usage:
    python main.py <command>

Commands:
    detect          Run real-time anomaly detection (Ctrl+C to stop)
    collect-normal  Collect 10 minutes of normal behavior data
    collect-anomaly Collect anomaly data with CPU stress injection
    train           Train and select the best anomaly detection model

Workflow:
    1. python main.py collect-normal    # Collect normal data
    2. python main.py collect-anomaly   # Collect anomaly data
    3. python main.py train             # Train models
    4. python main.py detect            # Run detector

To simulate an attack during detection:
    Terminal 1: python main.py detect
    Terminal 2: python stress.py 30     # 30 seconds of CPU stress
""")


def main():
    ensure_directories()
    
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(0)
    
    command = sys.argv[1].lower()
    
    commands = {
        "detect": cmd_detect,
        "collect-normal": cmd_collect_normal,
        "collect-anomaly": cmd_collect_anomaly,
        "train": cmd_train,
    }
    
    if command in commands:
        commands[command]()
    else:
        print(f"Unknown command: {command}")
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()
