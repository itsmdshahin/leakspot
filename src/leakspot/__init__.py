"""LeakSpot: find data leakage in machine-learning code and data."""

from .audit import AuditConfig, AuditReport, audit
from .scanner import Finding, ScanResult, scan_file, scan_source

__all__ = ["scan_file", "scan_source", "Finding", "ScanResult",
           "audit", "AuditConfig", "AuditReport"]
__version__ = "0.1.0"
