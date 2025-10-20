#!/usr/bin/env python3
"""
Launcher script for the Barrel Linear Actuator Simulator.

This script provides an easy way to run the application from the project root.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from kinematics.main import main

if __name__ == "__main__":
    main()
