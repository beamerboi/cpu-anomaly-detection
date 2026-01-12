"""
Script to collect normal system behavior data.

Run this script during typical laptop usage (browsing, idle, light work)
to gather baseline normal behavior data for training the anomaly detector.
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

from monitor.system_monitor import SystemMonitor
from config import (
    NORMAL_DATA_DIR,
    SAMPLING_INTERVAL,
    LABEL_NORMAL,
    ensure_directories,
)


def collect_normal_data(
    duration_minutes: float = 10,
    output_filename: str = None,
    sampling_interval: float = SAMPLING_INTERVAL,
):
    """
    Collect normal behavior data from the system.
    
    Args:
        duration_minutes: How long to collect data in minutes.
        output_filename: Name of output CSV file (auto-generated if None).
        sampling_interval: Time between samples in seconds.
    """
    ensure_directories()
    
    # Generate filename if not provided
    if output_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"normal_{timestamp}.csv"
    
    output_path = NORMAL_DATA_DIR / output_filename
    
    print("=" * 60)
    print("Normal Data Collection")
    print("=" * 60)
    print(f"Output file: {output_path}")
    print(f"Duration: {duration_minutes} minutes")
    print(f"Sampling interval: {sampling_interval} seconds")
    print(f"Expected samples: ~{int(duration_minutes * 60 / sampling_interval)}")
    print()
    print("Instructions:")
    print("  - Use your laptop normally during data collection")
    print("  - Activities: browsing, reading, light work, idle")
    print("  - AVOID: heavy computations, video encoding, gaming")
    print()
    print("Press Ctrl+C to stop early.")
    print("-" * 60)
    
    monitor = SystemMonitor(sampling_interval=sampling_interval)
    
    start_time = time.time()
    duration_seconds = duration_minutes * 60
    
    try:
        samples_collected = monitor.collect_to_csv(
            output_path=output_path,
            duration_seconds=duration_seconds,
            label=LABEL_NORMAL,
        )
        
        elapsed = time.time() - start_time
        print()
        print("-" * 60)
        print(f"Collection complete!")
        print(f"  Samples collected: {samples_collected}")
        print(f"  Duration: {elapsed:.1f} seconds")
        print(f"  Output file: {output_path}")
        
    except KeyboardInterrupt:
        monitor.stop()
        elapsed = time.time() - start_time
        print()
        print("-" * 60)
        print(f"Collection interrupted by user.")
        print(f"  Duration: {elapsed:.1f} seconds")
        print(f"  Output file: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect normal system behavior data for anomaly detection training."
    )
    parser.add_argument(
        "-d", "--duration",
        type=float,
        default=10,
        help="Collection duration in minutes (default: 10)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output filename (auto-generated if not provided)",
    )
    parser.add_argument(
        "-i", "--interval",
        type=float,
        default=SAMPLING_INTERVAL,
        help=f"Sampling interval in seconds (default: {SAMPLING_INTERVAL})",
    )
    
    args = parser.parse_args()
    
    collect_normal_data(
        duration_minutes=args.duration,
        output_filename=args.output,
        sampling_interval=args.interval,
    )


if __name__ == "__main__":
    main()
