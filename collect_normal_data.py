"""
Script to collect normal system behavior data.

Run this script during typical laptop usage (browsing, idle, light work)
to gather baseline normal behavior data for training the anomaly detector.
"""

import sys
import time
from datetime import datetime

from monitor.system_monitor import SystemMonitor
from config import (
    NORMAL_DATA_DIR,
    SAMPLING_INTERVAL,
    LABEL_NORMAL,
    ensure_directories,
)


def collect_normal_data(duration_minutes: float = 10):
    """
    Collect normal behavior data from the system.
    
    Args:
        duration_minutes: How long to collect data in minutes.
    """
    ensure_directories()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"normal_{timestamp}.csv"
    output_path = NORMAL_DATA_DIR / output_filename
    
    print("=" * 60)
    print("Normal Data Collection")
    print("=" * 60)
    print(f"Output file: {output_path}")
    print(f"Duration: {duration_minutes} minutes")
    print(f"Sampling interval: {SAMPLING_INTERVAL} seconds")
    print(f"Expected samples: ~{int(duration_minutes * 60 / SAMPLING_INTERVAL)}")
    print()
    print("Instructions:")
    print("  - Use your laptop normally during data collection")
    print("  - Activities: browsing, reading, light work, idle")
    print("  - AVOID: heavy computations, video encoding, gaming")
    print()
    print("Press Ctrl+C to stop early.")
    print("-" * 60)
    
    monitor = SystemMonitor(sampling_interval=SAMPLING_INTERVAL)
    
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


if __name__ == "__main__":
    # Simple usage: python collect_normal_data.py [duration_in_minutes]
    duration = 10
    if len(sys.argv) > 1:
        try:
            duration = float(sys.argv[1])
        except ValueError:
            print("Usage: python collect_normal_data.py [duration_in_minutes]")
            sys.exit(1)
    
    collect_normal_data(duration_minutes=duration)
