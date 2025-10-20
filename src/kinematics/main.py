"""
Main entry point for the barrel linear actuator simulator.

This module handles:
- Application initialization
- Main event loop
- PyVista theme setup
"""

import sys
import os

# Add the parent directory to the path so we can import from the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyvista as pv
from PyQt5.QtWidgets import QApplication

from src.kinematics.ui import RigApp

def main():
    """Main entry point for the application."""
    
    # Set up PyVista theme
    pv.global_theme.smooth_shading = True
    
    # Create Qt application
    app = QApplication(sys.argv)
    
    # Create and show main window
    window = RigApp()
    window.show()
    
    # Start event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
