"""
Configuration constants for the Anomaly Detection System.
"""

import os
from pathlib import Path

# =============================================================================
# Path Configuration
# =============================================================================

# Base directory (project root)
BASE_DIR = Path(__file__).parent.absolute()

# Data directories
DATA_DIR = BASE_DIR / "data"
NORMAL_DATA_DIR = DATA_DIR / "normal"
ANOMALY_DATA_DIR = DATA_DIR / "anomaly"
TRAINING_SET_PATH = DATA_DIR / "training_set.csv"

# Model directory
MODELS_DIR = BASE_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

# Logs directory
LOGS_DIR = BASE_DIR / "logs"

# =============================================================================
# Monitoring Configuration
# =============================================================================

# Sampling interval in seconds
SAMPLING_INTERVAL = 1.0

# Number of CPU cores (will be detected at runtime)
CPU_COUNT = os.cpu_count() or 4

# =============================================================================
# Feature Configuration
# =============================================================================

# Features to collect from system monitoring
FEATURE_NAMES = [
    "cpu_percent",           # Overall CPU usage percentage
    "cpu_freq_current",      # Current CPU frequency (MHz)
    "memory_percent",        # Memory usage percentage
    "memory_available_gb",   # Available memory in GB
    "num_processes",         # Number of running processes
    "num_threads",           # Total number of threads
    "ctx_switches",          # Context switches per second
    "interrupts",            # Interrupts per second
    "disk_read_bytes",       # Disk read bytes per second
    "disk_write_bytes",      # Disk write bytes per second
    "net_bytes_sent",        # Network bytes sent per second
    "net_bytes_recv",        # Network bytes received per second
]

# Add per-core CPU usage features
for i in range(CPU_COUNT):
    FEATURE_NAMES.append(f"cpu_core_{i}_percent")

# =============================================================================
# Anomaly Injection Configuration
# =============================================================================

# Default number of stress worker processes (use all logical processors for real stress)
DEFAULT_STRESS_WORKERS = CPU_COUNT  # 12 on your system

# Default stress intensity (iterations per loop)
DEFAULT_STRESS_INTENSITY = 500000  # Increased for more CPU load

# =============================================================================
# Training Configuration
# =============================================================================

# Train/test split ratio
TEST_SIZE = 0.3

# Cross-validation folds
CV_FOLDS = 5

# Random seed for reproducibility
RANDOM_SEED = 42

# =============================================================================
# Labels
# =============================================================================

LABEL_NORMAL = 0
LABEL_ANOMALY = 1

# =============================================================================
# Runtime Detection Configuration
# =============================================================================

# Detection log file
DETECTION_LOG_FILE = LOGS_DIR / "detection_log.csv"

# Console colors (ANSI codes, colorama handles Windows compatibility)
COLOR_NORMAL = "\033[92m"   # Green
COLOR_ANOMALY = "\033[91m"  # Red
COLOR_RESET = "\033[0m"     # Reset

# =============================================================================
# Ensure directories exist
# =============================================================================

def ensure_directories():
    """Create necessary directories if they don't exist."""
    for directory in [DATA_DIR, NORMAL_DATA_DIR, ANOMALY_DATA_DIR, MODELS_DIR, LOGS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
