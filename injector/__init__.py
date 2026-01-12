"""
Anomaly injection module for simulating system stress.
"""

from .cpu_stress import CPUStressInjector, GradualStressInjector

__all__ = ["CPUStressInjector", "GradualStressInjector"]
