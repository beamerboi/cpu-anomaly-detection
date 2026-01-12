"""
CPU Stress Injector Module

Simulates CPU stress anomalies using multiprocessing to bypass
Python's GIL and achieve real CPU load across all cores.
"""

import time
import multiprocessing
from datetime import datetime
from typing import Optional, List
from pathlib import Path
import os
import signal

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DEFAULT_STRESS_WORKERS,
    DEFAULT_STRESS_INTENSITY,
    CPU_COUNT,
)


def _cpu_stress_worker(stop_event: multiprocessing.Event, intensity: int):
    """
    Worker process that performs CPU-intensive calculations.
    
    This runs in a separate process to bypass the GIL.
    
    Args:
        stop_event: Event to signal when to stop.
        intensity: Number of iterations per loop.
    """
    # Ignore keyboard interrupt in workers (let parent handle it)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    while not stop_event.is_set():
        # Heavy CPU-bound operations
        
        # 1. Mathematical computations (tight loop)
        total = 0
        for i in range(intensity):
            total += i * i * i
        
        # 2. Floating point operations
        x = 1.0
        for _ in range(intensity // 10):
            x = x * 1.0000001
            x = x / 1.0000001
        
        # Check stop event periodically
        if stop_event.is_set():
            break


def _matrix_stress_worker(stop_event: multiprocessing.Event, size: int = 200):
    """
    Worker process that performs matrix multiplication.
    
    Args:
        stop_event: Event to signal when to stop.
        size: Matrix size (NxN).
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    import random
    
    while not stop_event.is_set():
        # Create random matrices
        matrix_a = [[random.random() for _ in range(size)] for _ in range(size)]
        matrix_b = [[random.random() for _ in range(size)] for _ in range(size)]
        
        # Multiply matrices (O(n^3) operation)
        result = [[0.0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    result[i][j] += matrix_a[i][k] * matrix_b[k][j]
        
        if stop_event.is_set():
            break


def _prime_stress_worker(stop_event: multiprocessing.Event, limit: int = 10000):
    """
    Worker process that calculates prime numbers.
    
    Args:
        stop_event: Event to signal when to stop.
        limit: Find primes up to this number.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    while not stop_event.is_set():
        # Find all primes up to limit (CPU intensive)
        primes = []
        for num in range(2, limit):
            is_prime = True
            for i in range(2, int(num ** 0.5) + 1):
                if num % i == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(num)
        
        if stop_event.is_set():
            break


class CPUStressInjector:
    """
    Inject CPU stress anomalies using multiprocessing.
    
    Uses separate processes (not threads) to bypass Python's GIL
    and achieve real CPU load across all cores.
    """
    
    def __init__(
        self,
        num_workers: int = DEFAULT_STRESS_WORKERS,
        intensity: int = DEFAULT_STRESS_INTENSITY,
    ):
        """
        Initialize the CPU stress injector.
        
        Args:
            num_workers: Number of worker processes (default: CPU count).
            intensity: Iterations per loop (higher = more CPU).
        """
        self.num_workers = num_workers
        self.intensity = intensity
        self._running = False
        self._processes: List[multiprocessing.Process] = []
        self._stop_event: Optional[multiprocessing.Event] = None
        self._start_time: Optional[datetime] = None
        self._stop_time: Optional[datetime] = None
    
    def start(self, stress_type: str = "compute"):
        """
        Start the CPU stress injection.
        
        Args:
            stress_type: Type of stress - "compute", "matrix", or "prime"
        """
        if self._running:
            raise RuntimeError("Stress injection is already running")
        
        self._stop_event = multiprocessing.Event()
        self._running = True
        self._start_time = datetime.now()
        self._stop_time = None
        
        # Select worker function
        if stress_type == "matrix":
            worker_func = _matrix_stress_worker
            worker_args = (self._stop_event, 150)
        elif stress_type == "prime":
            worker_func = _prime_stress_worker
            worker_args = (self._stop_event, 50000)
        else:
            worker_func = _cpu_stress_worker
            worker_args = (self._stop_event, self.intensity)
        
        # Create and start worker processes
        for i in range(self.num_workers):
            p = multiprocessing.Process(
                target=worker_func,
                args=worker_args,
                daemon=True,
                name=f"CPUStress-{i}",
            )
            p.start()
            self._processes.append(p)
        
        print(f"[{self._start_time.strftime('%H:%M:%S')}] CPU stress STARTED: "
              f"{self.num_workers} processes, type={stress_type}")
    
    def stop(self):
        """Stop the CPU stress injection."""
        if not self._running:
            return
        
        # Signal all workers to stop
        if self._stop_event:
            self._stop_event.set()
        
        self._running = False
        self._stop_time = datetime.now()
        
        # Wait for workers to finish
        for p in self._processes:
            p.join(timeout=2.0)
            if p.is_alive():
                p.terminate()
                p.join(timeout=1.0)
        
        self._processes.clear()
        
        duration = (self._stop_time - self._start_time).total_seconds()
        print(f"[{self._stop_time.strftime('%H:%M:%S')}] CPU stress STOPPED. "
              f"Duration: {duration:.1f}s")
    
    def run_for_duration(self, duration_seconds: float, stress_type: str = "compute"):
        """
        Run CPU stress for a specified duration.
        
        Args:
            duration_seconds: How long to run the stress test.
            stress_type: Type of stress ("compute", "matrix", or "prime").
        """
        self.start(stress_type=stress_type)
        try:
            time.sleep(duration_seconds)
        finally:
            self.stop()
    
    @property
    def is_running(self) -> bool:
        """Check if stress injection is currently running."""
        return self._running
    
    @property
    def start_time(self) -> Optional[datetime]:
        """Get the start time of the current/last stress session."""
        return self._start_time
    
    @property
    def stop_time(self) -> Optional[datetime]:
        """Get the stop time of the last stress session."""
        return self._stop_time
    
    def get_injection_period(self) -> tuple:
        """Get the injection period timestamps."""
        return (self._start_time, self._stop_time)


class GradualStressInjector:
    """
    Inject gradually increasing CPU stress for varied anomaly patterns.
    """
    
    def __init__(self, max_workers: int = DEFAULT_STRESS_WORKERS):
        """
        Initialize with maximum number of workers.
        
        Args:
            max_workers: Maximum number of stress processes.
        """
        self.max_workers = max_workers
        self._stop_event: Optional[multiprocessing.Event] = None
        self._processes: List[multiprocessing.Process] = []
        self._running = False
    
    def run_gradient(
        self,
        total_duration: float,
        steps: int = 4,
        intensity: int = DEFAULT_STRESS_INTENSITY,
    ):
        """
        Run gradually increasing stress.
        
        Args:
            total_duration: Total duration in seconds.
            steps: Number of intensity steps.
            intensity: Base intensity.
        """
        step_duration = total_duration / steps
        workers_per_step = max(1, self.max_workers // steps)
        
        self._stop_event = multiprocessing.Event()
        self._running = True
        
        print(f"Starting gradient stress: {steps} steps, {step_duration:.1f}s each")
        
        try:
            for step in range(steps):
                num_workers = min((step + 1) * workers_per_step, self.max_workers)
                print(f"  Step {step + 1}: {num_workers} workers")
                
                # Add more workers
                while len(self._processes) < num_workers:
                    p = multiprocessing.Process(
                        target=_cpu_stress_worker,
                        args=(self._stop_event, intensity),
                        daemon=True,
                    )
                    p.start()
                    self._processes.append(p)
                
                time.sleep(step_duration)
        
        finally:
            self.stop()
    
    def stop(self):
        """Stop all stress processes."""
        if self._stop_event:
            self._stop_event.set()
        
        for p in self._processes:
            p.join(timeout=2.0)
            if p.is_alive():
                p.terminate()
        
        self._processes.clear()
        self._running = False
        print("Gradient stress stopped.")


def main():
    """Test the CPU stress injector."""
    print("=" * 60)
    print("CPU Stress Injector Test")
    print("=" * 60)
    print(f"CPU Cores: {CPU_COUNT // 2} physical, {CPU_COUNT} logical")
    print(f"Workers: {DEFAULT_STRESS_WORKERS}")
    print()
    
    print("Running 10-second CPU stress test...")
    print("Watch Task Manager - CPU should spike to 80-100%")
    print("-" * 60)
    
    injector = CPUStressInjector(
        num_workers=DEFAULT_STRESS_WORKERS,
        intensity=DEFAULT_STRESS_INTENSITY,
    )
    
    try:
        injector.run_for_duration(10.0, stress_type="compute")
    except KeyboardInterrupt:
        injector.stop()
        print("\nStopped by user.")
    
    print()
    print("=" * 60)
    print("Test complete!")


if __name__ == "__main__":
    # Required for Windows multiprocessing
    multiprocessing.freeze_support()
    main()
