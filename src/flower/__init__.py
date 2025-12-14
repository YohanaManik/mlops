"""
Flower Classification MLOps Pipeline

Main package untuk klasifikasi 5 jenis bunga dengan pipeline MLOps lengkap.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from pathlib import Path

# Package root path
PACKAGE_ROOT = Path(__file__).parent
PROJECT_ROOT = PACKAGE_ROOT.parent

# Default paths
CONFIG_PATH = PROJECT_ROOT / "configs"
DATA_PATH = PROJECT_ROOT / "data"
ARTIFACTS_PATH = PROJECT_ROOT / "artifacts"
LOGS_PATH = PROJECT_ROOT / "logs"

# Ensure directories exist
ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
LOGS_PATH.mkdir(parents=True, exist_ok=True)

__all__ = [
    "PACKAGE_ROOT",
    "PROJECT_ROOT",
    "CONFIG_PATH",
    "DATA_PATH",
    "ARTIFACTS_PATH",
    "LOGS_PATH",
]