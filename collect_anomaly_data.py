"""
Script to collect anomaly (CPU stress) data.

This script automatically injects CPU stress while collecting
system metrics, creating labeled anomaly data for training.
"""

import argparse
import time
import threading
from datetime import datetime
from pathlib import Path

from monitor.system_monitor import SystemMonitor
from injector.cpu_stress import CPUStressInjector
from config import (
    ANOMALY_DATA_DIR,
    SAMPLING_INTERVAL,
    LABEL_ANOMALY,
    LABEL_NORMAL,
    DEFAULT_STRESS_WORKERS,
    DEFAULT_STRESS_INTENSITY,
    CPU_COUNT,
    ensure_directories,
)


def collect_anomaly_data(
    stress_duration_seconds: float = 60,
    num_workers: int = DEFAULT_STRESS_WORKERS,
    intensity: int = DEFAULT_STRESS_INTENSITY,
    warmup_seconds: float = 5,
    cooldown_seconds: float = 5,
    output_filename: str = None,
    sampling_interval: float = SAMPLING_INTERVAL,
):
    """
    Collect anomaly data by injecting CPU stress.
    
    Args:
        stress_duration_seconds: How long to run the stress test.
        num_workers: Number of stress worker processes.
        intensity: Stress intensity (iterations per loop).
        warmup_seconds: Seconds of normal data before stress.
        cooldown_seconds: Seconds of normal data after stress.
        output_filename: Name of output CSV file.
        sampling_interval: Time between samples in seconds.
    """
    ensure_directories()
    
    # Generate filename if not provided
    if output_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"anomaly_{timestamp}.csv"
    
    output_path = ANOMALY_DATA_DIR / output_filename
    
    total_duration = warmup_seconds + stress_duration_seconds + cooldown_seconds
    
    print("=" * 60)
    print("Anomaly Data Collection (CPU Stress)")
    print("=" * 60)
    print(f"Output file: {output_path}")
    print(f"Stress configuration:")
    print(f"  - Workers: {num_workers} processes (CPU cores: {CPU_COUNT})")
    print(f"  - Intensity: {intensity}")
    print(f"  - Stress duration: {stress_duration_seconds}s")
    print(f"  - Warmup period: {warmup_seconds}s (normal data)")
    print(f"  - Cooldown period: {cooldown_seconds}s (normal data)")
    print(f"  - Total duration: {total_duration}s")
    print(f"Sampling interval: {sampling_interval}s")
    print()
    print("Data will be labeled automatically:")
    print(f"  - Warmup/Cooldown: label = {LABEL_NORMAL} (normal)")
    print(f"  - During stress: label = {LABEL_ANOMALY} (anomaly)")
    print("-" * 60)
    
    monitor = SystemMonitor(sampling_interval=sampling_interval)
    injector = CPUStressInjector(num_workers=num_workers, intensity=intensity)
    
    # Results storage
    results = {
        "warmup_samples": 0,
        "stress_samples": 0,
        "cooldown_samples": 0,
    }
    
    start_time = time.time()
    
    try:
        # Phase 1: Warmup (normal behavior)
        print(f"\n[Phase 1] Warmup - collecting normal data for {warmup_seconds}s...")
        results["warmup_samples"] = monitor.collect_to_csv(
            output_path=output_path,
            duration_seconds=warmup_seconds,
            label=LABEL_NORMAL,
            append=False,
        )
        
        # Phase 2: Stress injection (anomaly behavior)
        print(f"\n[Phase 2] Stress injection - running for {stress_duration_seconds}s...")
        injector.start()
        
        results["stress_samples"] = monitor.collect_to_csv(
            output_path=output_path,
            duration_seconds=stress_duration_seconds,
            label=LABEL_ANOMALY,
            append=True,
        )
        
        injector.stop()
        
        # Phase 3: Cooldown (normal behavior)
        print(f"\n[Phase 3] Cooldown - collecting normal data for {cooldown_seconds}s...")
        results["cooldown_samples"] = monitor.collect_to_csv(
            output_path=output_path,
            duration_seconds=cooldown_seconds,
            label=LABEL_NORMAL,
            append=True,
        )
        
        elapsed = time.time() - start_time
        total_samples = sum(results.values())
        
        print()
        print("-" * 60)
        print("Collection complete!")
        print(f"  Warmup samples (normal): {results['warmup_samples']}")
        print(f"  Stress samples (anomaly): {results['stress_samples']}")
        print(f"  Cooldown samples (normal): {results['cooldown_samples']}")
        print(f"  Total samples: {total_samples}")
        print(f"  Duration: {elapsed:.1f} seconds")
        print(f"  Output file: {output_path}")
        
    except KeyboardInterrupt:
        injector.stop()
        monitor.stop()
        print()
        print("-" * 60)
        print("Collection interrupted by user.")
        print(f"  Output file: {output_path}")


def collect_intermittent_stress(
    total_duration_minutes: float = 10,
    stress_interval_seconds: float = 30,
    stress_duration_seconds: float = 15,
    num_workers: int = DEFAULT_STRESS_WORKERS,
    intensity: int = DEFAULT_STRESS_INTENSITY,
    output_filename: str = None,
    sampling_interval: float = SAMPLING_INTERVAL,
):
    """
    Collect data with intermittent stress periods (more realistic anomalies).
    
    This creates a dataset with alternating normal and anomaly periods,
    which is more representative of real-world anomaly patterns.
    
    Args:
        total_duration_minutes: Total collection time.
        stress_interval_seconds: Time between stress periods.
        stress_duration_seconds: Duration of each stress period.
        num_workers: Number of stress worker processes.
        intensity: Stress intensity.
        output_filename: Name of output CSV file.
        sampling_interval: Time between samples.
    """
    ensure_directories()
    
    if output_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"anomaly_intermittent_{timestamp}.csv"
    
    output_path = ANOMALY_DATA_DIR / output_filename
    
    total_duration_seconds = total_duration_minutes * 60
    
    print("=" * 60)
    print("Intermittent Anomaly Data Collection")
    print("=" * 60)
    print(f"Output file: {output_path}")
    print(f"Total duration: {total_duration_minutes} minutes")
    print(f"Stress pattern: {stress_duration_seconds}s stress every {stress_interval_seconds}s")
    print(f"Workers: {num_workers} processes, Intensity: {intensity}")
    print("-" * 60)
    
    monitor = SystemMonitor(sampling_interval=sampling_interval)
    injector = CPUStressInjector(num_workers=num_workers, intensity=intensity)
    
    start_time = time.time()
    first_write = True
    normal_samples = 0
    anomaly_samples = 0
    
    try:
        while time.time() - start_time < total_duration_seconds:
            # Normal period
            elapsed = time.time() - start_time
            normal_duration = min(
                stress_interval_seconds - stress_duration_seconds,
                total_duration_seconds - elapsed
            )
            
            if normal_duration > 0:
                print(f"\n[{elapsed:.0f}s] Normal period ({normal_duration:.0f}s)...")
                samples = monitor.collect_to_csv(
                    output_path=output_path,
                    duration_seconds=normal_duration,
                    label=LABEL_NORMAL,
                    append=not first_write,
                )
                normal_samples += samples
                first_write = False
            
            # Check if we have time for stress
            elapsed = time.time() - start_time
            if elapsed >= total_duration_seconds:
                break
            
            # Stress period
            stress_time = min(stress_duration_seconds, total_duration_seconds - elapsed)
            if stress_time > 0:
                print(f"[{elapsed:.0f}s] STRESS period ({stress_time:.0f}s)...")
                injector.start()
                samples = monitor.collect_to_csv(
                    output_path=output_path,
                    duration_seconds=stress_time,
                    label=LABEL_ANOMALY,
                    append=True,
                )
                anomaly_samples += samples
                injector.stop()
        
        print()
        print("-" * 60)
        print("Collection complete!")
        print(f"  Normal samples: {normal_samples}")
        print(f"  Anomaly samples: {anomaly_samples}")
        print(f"  Total samples: {normal_samples + anomaly_samples}")
        print(f"  Output file: {output_path}")
        
    except KeyboardInterrupt:
        injector.stop()
        monitor.stop()
        print()
        print("Collection interrupted by user.")


def main():
    parser = argparse.ArgumentParser(
        description="Collect anomaly (CPU stress) data for training."
    )
    
    subparsers = parser.add_subparsers(dest="mode", help="Collection mode")
    
    # Single stress mode
    single_parser = subparsers.add_parser(
        "single",
        help="Single continuous stress period"
    )
    single_parser.add_argument(
        "-d", "--duration",
        type=float,
        default=60,
        help="Stress duration in seconds (default: 60)",
    )
    single_parser.add_argument(
        "-w", "--workers",
        type=int,
        default=DEFAULT_STRESS_WORKERS,
        help=f"Number of stress worker processes (default: {DEFAULT_STRESS_WORKERS})",
    )
    single_parser.add_argument(
        "--intensity",
        type=int,
        default=DEFAULT_STRESS_INTENSITY,
        help=f"Stress intensity (default: {DEFAULT_STRESS_INTENSITY})",
    )
    single_parser.add_argument(
        "-w", "--warmup",
        type=float,
        default=5,
        help="Warmup period in seconds (default: 5)",
    )
    single_parser.add_argument(
        "-c", "--cooldown",
        type=float,
        default=5,
        help="Cooldown period in seconds (default: 5)",
    )
    single_parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output filename",
    )
    
    # Intermittent stress mode
    inter_parser = subparsers.add_parser(
        "intermittent",
        help="Intermittent stress periods"
    )
    inter_parser.add_argument(
        "-d", "--duration",
        type=float,
        default=10,
        help="Total duration in minutes (default: 10)",
    )
    inter_parser.add_argument(
        "--stress-interval",
        type=float,
        default=30,
        help="Seconds between stress periods (default: 30)",
    )
    inter_parser.add_argument(
        "--stress-duration",
        type=float,
        default=15,
        help="Duration of each stress period in seconds (default: 15)",
    )
    inter_parser.add_argument(
        "-w", "--workers",
        type=int,
        default=DEFAULT_STRESS_WORKERS,
        help=f"Number of stress worker processes (default: {DEFAULT_STRESS_WORKERS})",
    )
    inter_parser.add_argument(
        "--intensity",
        type=int,
        default=DEFAULT_STRESS_INTENSITY,
        help=f"Stress intensity (default: {DEFAULT_STRESS_INTENSITY})",
    )
    inter_parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output filename",
    )
    
    args = parser.parse_args()
    
    if args.mode == "single" or args.mode is None:
        # Default to single mode
        if args.mode is None:
            # Use defaults
            collect_anomaly_data()
        else:
            collect_anomaly_data(
                stress_duration_seconds=args.duration,
                num_workers=args.workers,
                intensity=args.intensity,
                warmup_seconds=args.warmup,
                cooldown_seconds=args.cooldown,
                output_filename=args.output,
            )
    elif args.mode == "intermittent":
        collect_intermittent_stress(
            total_duration_minutes=args.duration,
            stress_interval_seconds=args.stress_interval,
            stress_duration_seconds=args.stress_duration,
            num_workers=args.workers,
            intensity=args.intensity,
            output_filename=args.output,
        )


if __name__ == "__main__":
    main()
