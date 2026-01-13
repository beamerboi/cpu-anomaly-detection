#!/usr/bin/env python3
"""
CPU Stress Simulator

Run this script to simulate a CPU stress attack while the detector is monitoring.
"""

import sys
import time
import multiprocessing

from injector.cpu_stress import CPUStressInjector
from config import CPU_COUNT, DEFAULT_STRESS_WORKERS


def run_stress(duration: float = 30):
    """
    Run CPU stress for specified duration.
    
    Args:
        duration: Duration in seconds (default: 30).
    """
    print("=" * 60)
    print("CPU STRESS SIMULATOR")
    print("=" * 60)
    print(f"Duration: {duration} seconds")
    print(f"Workers: {DEFAULT_STRESS_WORKERS} processes")
    print(f"CPU cores: {CPU_COUNT} logical processors")
    print()
    print("TIP: Run 'python main.py detect' in another terminal")
    print("     to see if the detector catches this anomaly!")
    print("-" * 60)
    
    injector = CPUStressInjector(num_workers=DEFAULT_STRESS_WORKERS)
    
    try:
        injector.start()
        
        # Show countdown
        for remaining in range(int(duration), 0, -1):
            print(f"\r  Stressing... {remaining}s remaining   ", end="", flush=True)
            time.sleep(1)
        
        print("\r  Stress complete!                    ")
        
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    finally:
        injector.stop()
    
    print("-" * 60)
    print("Stress simulation finished.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    # Simple usage: python stress.py [duration_in_seconds]
    duration = 30
    if len(sys.argv) > 1:
        try:
            duration = float(sys.argv[1])
        except ValueError:
            print("Usage: python stress.py [duration_in_seconds]")
            print("Example: python stress.py 60")
            sys.exit(1)
    
    run_stress(duration)
