"""
Kinematics simulation package for barrel linear actuator systems.

This package provides:
- Geometric calculations and 3D modeling
- Kinematic analysis and feasibility checking
- PyQt5-based user interface
- Configuration management
- Main application entry point
"""

from .geometry import *
from .kinematics import KinematicsAnalyzer
from .config import ConfigManager
from .analysis import AnalysisManager
from .visualization import VisualizationManager
from .setup_manager import SetupManager
from .ui import RigApp
from .main import main

__version__ = "1.0.0"
__all__ = [
    "KinematicsAnalyzer",
    "ConfigManager",
    "AnalysisManager",
    "VisualizationManager", 
    "SetupManager",
    "RigApp",
    "main"
]