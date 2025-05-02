#!/usr/bin/env python3
"""
Utility functions for configuration loading and reproducibility
"""

import os
import yaml
import random
import numpy as np
import torch


def load_config(config_path):
    """Load parameters from YAML config file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found. Please provide a valid config file.")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def set_seed(seed):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Integer seed value for random number generators
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)