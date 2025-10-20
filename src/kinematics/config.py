"""
Configuration management for the barrel linear actuator simulator.

This module handles:
- Loading and saving setup configurations
- INI file parsing and writing
- Setup validation and management
"""

import configparser
import os

# Import geometry constants directly to avoid relative import issues
from .geometry import SETUP_KEYS


class ConfigManager:
    """Manages configuration loading, saving, and validation."""
    
    def __init__(self):
        self.setups = {}
        self.current_setups_path = None
    
    def read_setups_ini(self, path):
        """
        Parse an INI file with multiple sections.
        Returns a dict: {section_name: {key: float, ...}, ...}
        Unrecognized keys are ignored; comments allowed.
        """
        cp = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
        cp.optionxform = str  # preserve case
        
        if not cp.read(path, encoding="utf-8"):
            raise RuntimeError("Unable to read file or empty file")
        
        valid_keys = set(SETUP_KEYS)
        setups = {}
        for section in cp.sections():
            vals = {}
            for k in valid_keys:
                if cp.has_option(section, k):
                    try:
                        vals[k] = float(cp.get(section, k))
                    except ValueError:
                        raise ValueError(f"Section [{section}] key '{k}' must be a number")
            if vals:
                setups[section] = vals
        
        if not setups:
            raise RuntimeError("No valid sections/keys found")
        return setups
    
    def write_setups_ini(self, path, setups_dict):
        """
        Write all setups to an INI file.
        setups_dict: {section_name: {key: float, ...}, ...}
        Unknown keys are ignored. Sections with no valid keys are skipped.
        """
        valid_keys = SETUP_KEYS  # use the unified list
        
        cp = configparser.ConfigParser()
        cp.optionxform = str  # preserve key case
        
        # Merge with existing file (don't drop unrelated sections)
        if os.path.exists(path):
            try:
                cp.read(path, encoding="utf-8")
            except Exception:
                cp = configparser.ConfigParser()
                cp.optionxform = str
        
        for section, vals in setups_dict.items():
            if section not in cp.sections():
                cp.add_section(section)
            for k in valid_keys:
                if k in vals and vals[k] is not None:
                    cp.set(section, k, str(float(vals[k])))
        
        with open(path, "w", encoding="utf-8") as f:
            cp.write(f)
    
    def load_setups(self, path):
        """Load setups from a file and update internal state."""
        setups = self.read_setups_ini(path)
        self.setups = setups
        self.current_setups_path = path
        return setups
    
    def save_setups(self, path, setups_dict=None):
        """Save setups to a file."""
        if setups_dict is None:
            setups_dict = self.setups
        
        self.write_setups_ini(path, setups_dict)
        self.current_setups_path = path
    
    def add_setup(self, name, setup_dict):
        """Add a new setup to the current collection."""
        self.setups[name] = setup_dict
    
    def get_setup(self, name):
        """Get a setup by name."""
        return self.setups.get(name)
    
    def get_setup_names(self):
        """Get all setup names."""
        return sorted(self.setups.keys())
    
    def validate_setup(self, setup_dict):
        """Validate that a setup dictionary contains valid values."""
        if not isinstance(setup_dict, dict):
            return False
        
        for key in SETUP_KEYS:
            if key in setup_dict:
                try:
                    float(setup_dict[key])
                except (ValueError, TypeError):
                    return False
        
        return True
