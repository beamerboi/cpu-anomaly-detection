"""
Script to collect anomaly (CPU stress) data.

This script automatically injects CPU stress while collecting
system metrics, creating labeled anomaly data for training.
"""

import sys
import time
from datetime import datetime

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
    warmup_seconds: float = 5,
    cooldown_seconds: float = 5,
):
    """
    Collect anomaly data by injecting CPU stress.
    
    Args:
        stress_duration_seconds: How long to run the stress test.
        warmup_seconds: Seconds of normal data before stress.
        cooldown_seconds: Seconds of normal data after stress.
    """
    ensure_directories()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"anomaly_{timestamp}.csv"
    output_path = ANOMALY_DATA_DIR / output_filename
    
    total_duration = warmup_seconds + stress_duration_seconds + cooldown_seconds
    
    print("=" * 60)
    print("Anomaly Data Collection (CPU Stress)")
    print("=" * 60)
    print(f"Output file: {output_path}")
    print(f"Stress configuration:")
    print(f"  - Workers: {DEFAULT_STRESS_WORKERS} processes (CPU cores: {CPU_COUNT})")
    print(f"  - Intensity: {DEFAULT_STRESS_INTENSITY}")
    print(f"  - Stress duration: {stress_duration_seconds}s")
    print(f"  - Warmup period: {warmup_seconds}s (normal data)")
    print(f"  - Cooldown period: {cooldown_seconds}s (normal data)")
    print(f"  - Total duration: {total_duration}s")
    print(f"Sampling interval: {SAMPLING_INTERVAL}s")
    print()
    print("Data will be labeled automatically:")
    print(f"  - Warmup/Cooldown: label = {LABEL_NORMAL} (normal)")
    print(f"  - During stress: label = {LABEL_ANOMALY} (anomaly)")
    print("-" * 60)
    
    monitor = SystemMonitor(sampling_interval=SAMPLING_INTERVAL)
    injector = CPUStressInjector(num_workers=DEFAULT_STRESS_WORKERS, intensity=DEFAULT_STRESS_INTENSITY)
    
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


if __name__ == "__main__":
    # Simple usage: python collect_anomaly_data.py [stress_duration_seconds]
    stress_duration = 60
    if len(sys.argv) > 1:
        try:
            stress_duration = float(sys.argv[1])
        except ValueError:
            print("Usage: python collect_anomaly_data.py [stress_duration_seconds]")
            sys.exit(1)
    
    collect_anomaly_data(stress_duration_seconds=stress_duration)
