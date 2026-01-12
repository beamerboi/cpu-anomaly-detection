"""
Runtime Anomaly Detector Module

Real-time anomaly detection using trained models and live system monitoring.
"""

import time
import csv
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
import sys

import numpy as np
import joblib

# Windows console color support
try:
    from colorama import init, Fore, Style
    init()
    COLOR_SUPPORT = True
except ImportError:
    COLOR_SUPPORT = False
    class Fore:
        GREEN = ""
        RED = ""
        YELLOW = ""
        RESET = ""
    class Style:
        BRIGHT = ""
        RESET_ALL = ""

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    BEST_MODEL_PATH,
    SCALER_PATH,
    DETECTION_LOG_FILE,
    FEATURE_NAMES,
    SAMPLING_INTERVAL,
    LABEL_NORMAL,
    LABEL_ANOMALY,
    ensure_directories,
)
from monitor.system_monitor import SystemMonitor


class RuntimeDetector:
    """
    Real-time anomaly detector for system monitoring.
    
    Combines the system monitor with a trained ML model
    to detect anomalies in real-time.
    """
    
    def __init__(
        self,
        model_path: Path = BEST_MODEL_PATH,
        scaler_path: Path = SCALER_PATH,
        sampling_interval: float = SAMPLING_INTERVAL,
    ):
        """
        Initialize the runtime detector.
        
        Args:
            model_path: Path to the trained model.
            scaler_path: Path to the fitted scaler.
            sampling_interval: Time between samples in seconds.
        """
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.sampling_interval = sampling_interval
        
        self.model = None
        self.scaler = None
        self.monitor = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Statistics
        self.stats = {
            "total_samples": 0,
            "normal_count": 0,
            "anomaly_count": 0,
            "start_time": None,
        }
        
        # Determine if model is unsupervised
        self._is_unsupervised = False
    
    def load_model(self):
        """Load the trained model and scaler."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                "Please train the model first using: python train.py"
            )
        
        print(f"Loading model from: {self.model_path}")
        self.model = joblib.load(self.model_path)
        
        # Check if unsupervised
        model_type = type(self.model).__name__
        self._is_unsupervised = model_type in ["IsolationForest", "OneClassSVM", "LocalOutlierFactor"]
        
        if self.scaler_path.exists():
            print(f"Loading scaler from: {self.scaler_path}")
            self.scaler = joblib.load(self.scaler_path)
        else:
            print("Warning: Scaler not found, using unscaled features")
            self.scaler = None
        
        print(f"Model type: {model_type}")
        print(f"Unsupervised: {self._is_unsupervised}")
    
    def predict(self, features: np.ndarray) -> int:
        """
        Make a prediction on feature vector.
        
        Args:
            features: Feature vector (1D array).
        
        Returns:
            0 for normal, 1 for anomaly.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Reshape for single sample
        X = features.reshape(1, -1)
        
        # Scale if scaler is available
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        # Predict
        if self._is_unsupervised:
            # Unsupervised models return -1 for anomaly, 1 for normal
            pred = self.model.predict(X)[0]
            return LABEL_ANOMALY if pred == -1 else LABEL_NORMAL
        else:
            return int(self.model.predict(X)[0])
    
    def _format_output(
        self,
        timestamp: str,
        prediction: int,
        key_metrics: dict,
    ) -> str:
        """Format output string with colors."""
        if prediction == LABEL_ANOMALY:
            status = f"{Fore.RED}{Style.BRIGHT}ANOMALY{Style.RESET_ALL}"
        else:
            status = f"{Fore.GREEN}NORMAL{Style.RESET_ALL}"
        
        output = (
            f"[{timestamp}] {status} | "
            f"CPU: {key_metrics['cpu_percent']:5.1f}% | "
            f"MEM: {key_metrics['memory_percent']:5.1f}% | "
            f"Procs: {key_metrics['num_processes']:4d} | "
            f"CtxSw: {key_metrics['ctx_switches']:8.0f}/s"
        )
        
        return output
    
    def _log_detection(
        self,
        log_file: Path,
        timestamp: str,
        prediction: int,
        features: np.ndarray,
    ):
        """Log detection result to CSV file."""
        file_exists = log_file.exists()
        
        with open(log_file, 'a', newline='', encoding='utf-8') as f:
            fieldnames = ["timestamp", "prediction"] + FEATURE_NAMES
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            row = {"timestamp": timestamp, "prediction": prediction}
            for i, name in enumerate(FEATURE_NAMES):
                if i < len(features):
                    row[name] = features[i]
                else:
                    row[name] = 0.0
            
            writer.writerow(row)
    
    def run(
        self,
        duration_seconds: Optional[float] = None,
        log_to_file: bool = True,
        log_path: Path = DETECTION_LOG_FILE,
        verbose: bool = True,
    ) -> dict:
        """
        Run the detector in real-time.
        
        Args:
            duration_seconds: How long to run (None for indefinite).
            log_to_file: If True, log results to CSV file.
            log_path: Path to log file.
            verbose: If True, print to console.
        
        Returns:
            Statistics dictionary.
        """
        ensure_directories()
        
        if self.model is None:
            self.load_model()
        
        self.monitor = SystemMonitor(sampling_interval=self.sampling_interval)
        
        # Reset statistics
        self.stats = {
            "total_samples": 0,
            "normal_count": 0,
            "anomaly_count": 0,
            "start_time": datetime.now(),
        }
        
        if verbose:
            print("\n" + "=" * 70)
            print("REAL-TIME ANOMALY DETECTION")
            print("=" * 70)
            print(f"Sampling interval: {self.sampling_interval}s")
            print(f"Log file: {log_path if log_to_file else 'disabled'}")
            print(f"Press Ctrl+C to stop")
            print("-" * 70)
        
        self._running = True
        start_time = time.time()
        
        # Initial CPU measurement
        import psutil
        psutil.cpu_percent(interval=None)
        time.sleep(0.1)
        
        try:
            while self._running:
                # Collect sample
                sample = self.monitor.collect_sample()
                
                # Extract features
                features = np.array([
                    sample.get(name, 0.0) for name in FEATURE_NAMES
                ])
                
                # Handle any NaN or inf values
                features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)
                
                # Make prediction
                prediction = self.predict(features)
                
                # Update statistics
                self.stats["total_samples"] += 1
                if prediction == LABEL_ANOMALY:
                    self.stats["anomaly_count"] += 1
                else:
                    self.stats["normal_count"] += 1
                
                # Prepare key metrics for display
                key_metrics = {
                    "cpu_percent": sample.get("cpu_percent", 0),
                    "memory_percent": sample.get("memory_percent", 0),
                    "num_processes": int(sample.get("num_processes", 0)),
                    "ctx_switches": sample.get("ctx_switches", 0),
                }
                
                # Console output
                if verbose:
                    output = self._format_output(
                        sample["timestamp"],
                        prediction,
                        key_metrics,
                    )
                    print(output)
                
                # Log to file
                if log_to_file:
                    self._log_detection(log_path, sample["timestamp"], prediction, features)
                
                # Check duration
                if duration_seconds is not None:
                    if time.time() - start_time >= duration_seconds:
                        break
                
                # Wait for next sample
                time.sleep(self.sampling_interval)
        
        except KeyboardInterrupt:
            if verbose:
                print("\n\nDetection stopped by user.")
        
        finally:
            self._running = False
            self.stats["end_time"] = datetime.now()
        
        # Print final statistics
        if verbose:
            self._print_statistics()
        
        return self.stats
    
    def _print_statistics(self):
        """Print detection statistics."""
        print("\n" + "=" * 70)
        print("DETECTION SUMMARY")
        print("=" * 70)
        print(f"Total samples: {self.stats['total_samples']}")
        print(f"Normal: {self.stats['normal_count']} ({100 * self.stats['normal_count'] / max(1, self.stats['total_samples']):.1f}%)")
        print(f"Anomaly: {self.stats['anomaly_count']} ({100 * self.stats['anomaly_count'] / max(1, self.stats['total_samples']):.1f}%)")
        
        if self.stats.get("end_time") and self.stats.get("start_time"):
            duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()
            print(f"Duration: {duration:.1f} seconds")
    
    def start_background(
        self,
        log_to_file: bool = True,
        log_path: Path = DETECTION_LOG_FILE,
    ):
        """Start detection in background thread."""
        if self._running:
            raise RuntimeError("Detector is already running")
        
        self._thread = threading.Thread(
            target=self.run,
            kwargs={
                "duration_seconds": None,
                "log_to_file": log_to_file,
                "log_path": log_path,
                "verbose": False,
            },
            daemon=True,
        )
        self._thread.start()
    
    def stop(self):
        """Stop background detection."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
    
    @property
    def is_running(self) -> bool:
        """Check if detector is running."""
        return self._running


def main():
    """Run the detector from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Real-time anomaly detection for system monitoring."
    )
    parser.add_argument(
        "-d", "--duration",
        type=float,
        default=None,
        help="Detection duration in seconds (default: run until Ctrl+C)",
    )
    parser.add_argument(
        "-i", "--interval",
        type=float,
        default=SAMPLING_INTERVAL,
        help=f"Sampling interval in seconds (default: {SAMPLING_INTERVAL})",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Don't log results to file",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet mode (no console output)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(BEST_MODEL_PATH),
        help=f"Path to trained model (default: {BEST_MODEL_PATH})",
    )
    
    args = parser.parse_args()
    
    detector = RuntimeDetector(
        model_path=Path(args.model),
        sampling_interval=args.interval,
    )
    
    try:
        detector.run(
            duration_seconds=args.duration,
            log_to_file=not args.no_log,
            verbose=not args.quiet,
        )
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease train the model first:")
        print("  1. Collect data: python collect_normal_data.py && python collect_anomaly_data.py")
        print("  2. Train model: python train.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
