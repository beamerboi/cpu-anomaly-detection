#!/usr/bin/env python3
"""
CPU Stress Simulator

Run this script to simulate a CPU stress attack while the detector is monitoring.
This allows you to test if the anomaly detector correctly identifies the attack.

Usage:
    python stress.py                    # 30 seconds of stress (default)
    python stress.py -d 60              # 60 seconds of stress
    python stress.py -w 6               # Use only 6 workers (half CPU)
    python stress.py --type matrix      # Use matrix multiplication stress
    python stress.py --gradient         # Gradually increasing stress
"""

import argparse
import sys
import time
import multiprocessing

from injector.cpu_stress import CPUStressInjector, GradualStressInjector
from config import CPU_COUNT, DEFAULT_STRESS_WORKERS


def run_stress(
    duration: float = 30,
    num_workers: int = DEFAULT_STRESS_WORKERS,
    stress_type: str = "compute",
):
    """
    Run CPU stress for specified duration.
    
    Args:
        duration: Duration in seconds.
        num_workers: Number of worker processes.
        stress_type: Type of stress (compute, matrix, prime).
    """
    print("=" * 60)
    print("CPU STRESS SIMULATOR")
    print("=" * 60)
    print(f"Duration: {duration} seconds")
    print(f"Workers: {num_workers} processes")
    print(f"Stress type: {stress_type}")
    print(f"CPU cores: {CPU_COUNT} logical processors")
    print()
    print("TIP: Run 'python main.py detect' in another terminal")
    print("     to see if the detector catches this anomaly!")
    print("-" * 60)
    
    injector = CPUStressInjector(num_workers=num_workers)
    
    try:
        injector.start(stress_type=stress_type)
        
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


def run_gradient_stress(duration: float = 60, num_workers: int = DEFAULT_STRESS_WORKERS):
    """
    Run gradually increasing CPU stress.
    
    Args:
        duration: Total duration in seconds.
        num_workers: Maximum number of workers.
    """
    print("=" * 60)
    print("GRADIENT CPU STRESS SIMULATOR")
    print("=" * 60)
    print(f"Duration: {duration} seconds")
    print(f"Max workers: {num_workers}")
    print("Pattern: Gradually increasing CPU load in 4 steps")
    print("-" * 60)
    
    injector = GradualStressInjector(max_workers=num_workers)
    
    try:
        injector.run_gradient(total_duration=duration, steps=4)
    except KeyboardInterrupt:
        injector.stop()
        print("\nStopped by user.")
    
    print("-" * 60)
    print("Gradient stress finished.")


def run_burst_stress(
    num_bursts: int = 5,
    burst_duration: float = 10,
    pause_duration: float = 5,
    num_workers: int = DEFAULT_STRESS_WORKERS,
):
    """
    Run intermittent burst stress (simulates sporadic attacks).
    
    Args:
        num_bursts: Number of stress bursts.
        burst_duration: Duration of each burst in seconds.
        pause_duration: Pause between bursts in seconds.
        num_workers: Number of workers per burst.
    """
    print("=" * 60)
    print("BURST CPU STRESS SIMULATOR")
    print("=" * 60)
    print(f"Bursts: {num_bursts}")
    print(f"Burst duration: {burst_duration}s")
    print(f"Pause between: {pause_duration}s")
    print(f"Workers: {num_workers}")
    print("-" * 60)
    
    injector = CPUStressInjector(num_workers=num_workers)
    
    try:
        for i in range(num_bursts):
            print(f"\n[Burst {i+1}/{num_bursts}] Starting stress...")
            injector.start()
            time.sleep(burst_duration)
            injector.stop()
            
            if i < num_bursts - 1:
                print(f"[Pause] Waiting {pause_duration}s...")
                time.sleep(pause_duration)
        
    except KeyboardInterrupt:
        injector.stop()
        print("\n\nStopped by user.")
    
    print("-" * 60)
    print("Burst stress finished.")


def main():
    parser = argparse.ArgumentParser(
        description="CPU Stress Simulator - Test your anomaly detector!",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python stress.py                     # 30s default stress
    python stress.py -d 60               # 60 seconds
    python stress.py -w 6                # Use 6 workers (partial CPU)
    python stress.py --type matrix       # Matrix multiplication stress
    python stress.py --gradient -d 60    # Gradual increase over 60s
    python stress.py --burst             # 5 bursts of 10s each
        """
    )
    
    parser.add_argument(
        "-d", "--duration",
        type=float,
        default=30,
        help="Stress duration in seconds (default: 30)",
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=DEFAULT_STRESS_WORKERS,
        help=f"Number of worker processes (default: {DEFAULT_STRESS_WORKERS})",
    )
    parser.add_argument(
        "--type",
        choices=["compute", "matrix", "prime"],
        default="compute",
        help="Type of CPU stress (default: compute)",
    )
    parser.add_argument(
        "--gradient",
        action="store_true",
        help="Use gradually increasing stress pattern",
    )
    parser.add_argument(
        "--burst",
        action="store_true",
        help="Use burst/intermittent stress pattern",
    )
    parser.add_argument(
        "--bursts",
        type=int,
        default=5,
        help="Number of bursts for --burst mode (default: 5)",
    )
    parser.add_argument(
        "--burst-duration",
        type=float,
        default=10,
        help="Duration of each burst in seconds (default: 10)",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=5,
        help="Pause between bursts in seconds (default: 5)",
    )
    
    args = parser.parse_args()
    
    if args.gradient:
        run_gradient_stress(
            duration=args.duration,
            num_workers=args.workers,
        )
    elif args.burst:
        run_burst_stress(
            num_bursts=args.bursts,
            burst_duration=args.burst_duration,
            pause_duration=args.pause,
            num_workers=args.workers,
        )
    else:
        run_stress(
            duration=args.duration,
            num_workers=args.workers,
            stress_type=args.type,
        )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
