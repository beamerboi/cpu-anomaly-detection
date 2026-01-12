"""
System Monitor Module

Collects system performance indicators using psutil library.
Outputs data to CSV format for machine learning analysis.
"""

import time
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import threading

import psutil

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FEATURE_NAMES,
    SAMPLING_INTERVAL,
    CPU_COUNT,
    ensure_directories,
)


class SystemMonitor:
    """
    Monitor system performance indicators and save to CSV.
    
    Collects CPU, memory, disk, network, and process metrics
    at configurable intervals.
    """
    
    def __init__(self, sampling_interval: float = SAMPLING_INTERVAL):
        """
        Initialize the system monitor.
        
        Args:
            sampling_interval: Time between samples in seconds (default: 1.0)
        """
        self.sampling_interval = sampling_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Previous values for calculating rates
        self._prev_ctx_switches = 0
        self._prev_interrupts = 0
        self._prev_disk_read = 0
        self._prev_disk_write = 0
        self._prev_net_sent = 0
        self._prev_net_recv = 0
        self._prev_time = time.time()
        
        # Initialize previous values
        self._initialize_counters()
    
    def _initialize_counters(self):
        """Initialize counter values for rate calculations."""
        try:
            cpu_stats = psutil.cpu_stats()
            self._prev_ctx_switches = cpu_stats.ctx_switches
            self._prev_interrupts = cpu_stats.interrupts
        except Exception:
            pass
        
        try:
            disk_io = psutil.disk_io_counters()
            if disk_io:
                self._prev_disk_read = disk_io.read_bytes
                self._prev_disk_write = disk_io.write_bytes
        except Exception:
            pass
        
        try:
            net_io = psutil.net_io_counters()
            if net_io:
                self._prev_net_sent = net_io.bytes_sent
                self._prev_net_recv = net_io.bytes_recv
        except Exception:
            pass
        
        self._prev_time = time.time()
    
    def collect_sample(self) -> Dict[str, Any]:
        """
        Collect a single sample of system metrics.
        
        Returns:
            Dictionary containing all monitored metrics with their values.
        """
        current_time = time.time()
        time_delta = max(current_time - self._prev_time, 0.001)  # Avoid division by zero
        
        sample = {
            "timestamp": datetime.now().isoformat(),
        }
        
        # CPU metrics
        try:
            sample["cpu_percent"] = psutil.cpu_percent(interval=None)
            
            cpu_freq = psutil.cpu_freq()
            sample["cpu_freq_current"] = cpu_freq.current if cpu_freq else 0.0
            
            # Per-core CPU usage
            per_cpu = psutil.cpu_percent(interval=None, percpu=True)
            for i in range(CPU_COUNT):
                if i < len(per_cpu):
                    sample[f"cpu_core_{i}_percent"] = per_cpu[i]
                else:
                    sample[f"cpu_core_{i}_percent"] = 0.0
        except Exception as e:
            sample["cpu_percent"] = 0.0
            sample["cpu_freq_current"] = 0.0
            for i in range(CPU_COUNT):
                sample[f"cpu_core_{i}_percent"] = 0.0
        
        # Memory metrics
        try:
            mem = psutil.virtual_memory()
            sample["memory_percent"] = mem.percent
            sample["memory_available_gb"] = mem.available / (1024 ** 3)
        except Exception:
            sample["memory_percent"] = 0.0
            sample["memory_available_gb"] = 0.0
        
        # Process metrics
        try:
            sample["num_processes"] = len(psutil.pids())
            # Count total threads across all accessible processes
            total_threads = 0
            for proc in psutil.process_iter(['num_threads']):
                try:
                    total_threads += proc.info['num_threads'] or 0
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            sample["num_threads"] = total_threads
        except Exception:
            sample["num_processes"] = 0
            sample["num_threads"] = 0
        
        # CPU stats (context switches, interrupts) - calculate rate
        try:
            cpu_stats = psutil.cpu_stats()
            ctx_switches = cpu_stats.ctx_switches
            interrupts = cpu_stats.interrupts
            
            sample["ctx_switches"] = (ctx_switches - self._prev_ctx_switches) / time_delta
            sample["interrupts"] = (interrupts - self._prev_interrupts) / time_delta
            
            self._prev_ctx_switches = ctx_switches
            self._prev_interrupts = interrupts
        except Exception:
            sample["ctx_switches"] = 0.0
            sample["interrupts"] = 0.0
        
        # Disk I/O metrics - calculate rate
        try:
            disk_io = psutil.disk_io_counters()
            if disk_io:
                disk_read = disk_io.read_bytes
                disk_write = disk_io.write_bytes
                
                sample["disk_read_bytes"] = (disk_read - self._prev_disk_read) / time_delta
                sample["disk_write_bytes"] = (disk_write - self._prev_disk_write) / time_delta
                
                self._prev_disk_read = disk_read
                self._prev_disk_write = disk_write
            else:
                sample["disk_read_bytes"] = 0.0
                sample["disk_write_bytes"] = 0.0
        except Exception:
            sample["disk_read_bytes"] = 0.0
            sample["disk_write_bytes"] = 0.0
        
        # Network I/O metrics - calculate rate
        try:
            net_io = psutil.net_io_counters()
            if net_io:
                net_sent = net_io.bytes_sent
                net_recv = net_io.bytes_recv
                
                sample["net_bytes_sent"] = (net_sent - self._prev_net_sent) / time_delta
                sample["net_bytes_recv"] = (net_recv - self._prev_net_recv) / time_delta
                
                self._prev_net_sent = net_sent
                self._prev_net_recv = net_recv
            else:
                sample["net_bytes_sent"] = 0.0
                sample["net_bytes_recv"] = 0.0
        except Exception:
            sample["net_bytes_sent"] = 0.0
            sample["net_bytes_recv"] = 0.0
        
        self._prev_time = current_time
        
        return sample
    
    def get_feature_vector(self) -> List[float]:
        """
        Get current metrics as a feature vector for ML model.
        
        Returns:
            List of float values corresponding to FEATURE_NAMES.
        """
        sample = self.collect_sample()
        return [sample.get(feature, 0.0) for feature in FEATURE_NAMES]
    
    def collect_to_csv(
        self,
        output_path: Path,
        duration_seconds: Optional[float] = None,
        num_samples: Optional[int] = None,
        label: Optional[int] = None,
        append: bool = False,
    ) -> int:
        """
        Collect samples and write to CSV file.
        
        Args:
            output_path: Path to output CSV file.
            duration_seconds: How long to collect data (mutually exclusive with num_samples).
            num_samples: Number of samples to collect (mutually exclusive with duration_seconds).
            label: Optional label to add to each row (0=normal, 1=anomaly).
            append: If True, append to existing file; otherwise, overwrite.
        
        Returns:
            Number of samples collected.
        """
        ensure_directories()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine field names
        fieldnames = ["timestamp"] + FEATURE_NAMES
        if label is not None:
            fieldnames.append("label")
        
        mode = 'a' if append else 'w'
        file_exists = output_path.exists() and append
        
        samples_collected = 0
        start_time = time.time()
        
        # Initial CPU percent call to start measuring
        psutil.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None, percpu=True)
        
        with open(output_path, mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            self._running = True
            
            while self._running:
                sample = self.collect_sample()
                
                # Add label if provided
                if label is not None:
                    sample["label"] = label
                
                # Write only the fields we need
                row = {k: sample.get(k, 0.0) for k in fieldnames}
                writer.writerow(row)
                f.flush()  # Ensure data is written immediately
                
                samples_collected += 1
                
                # Check termination conditions
                if num_samples is not None and samples_collected >= num_samples:
                    break
                
                if duration_seconds is not None:
                    if time.time() - start_time >= duration_seconds:
                        break
                
                # Wait for next sample
                time.sleep(self.sampling_interval)
        
        return samples_collected
    
    def start_background_collection(
        self,
        output_path: Path,
        label: Optional[int] = None,
        append: bool = False,
    ):
        """
        Start collecting data in background thread.
        
        Args:
            output_path: Path to output CSV file.
            label: Optional label to add to each row.
            append: If True, append to existing file.
        """
        if self._running:
            raise RuntimeError("Monitor is already running")
        
        self._thread = threading.Thread(
            target=self.collect_to_csv,
            kwargs={
                "output_path": output_path,
                "duration_seconds": None,
                "num_samples": None,
                "label": label,
                "append": append,
            },
            daemon=True,
        )
        self._thread.start()
    
    def stop(self):
        """Stop background collection."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
    
    @property
    def is_running(self) -> bool:
        """Check if monitor is currently running."""
        return self._running


def main():
    """Test the system monitor by collecting a few samples."""
    print("System Monitor Test")
    print("=" * 50)
    
    monitor = SystemMonitor(sampling_interval=1.0)
    
    print("\nCollecting 5 samples...")
    print("-" * 50)
    
    for i in range(5):
        sample = monitor.collect_sample()
        print(f"\nSample {i + 1}:")
        print(f"  Timestamp: {sample['timestamp']}")
        print(f"  CPU: {sample['cpu_percent']:.1f}%")
        print(f"  Memory: {sample['memory_percent']:.1f}%")
        print(f"  Processes: {sample['num_processes']}")
        print(f"  Context Switches/s: {sample['ctx_switches']:.0f}")
        time.sleep(1)
    
    print("\n" + "=" * 50)
    print("Test complete!")


if __name__ == "__main__":
    main()
