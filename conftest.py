"""Pytest configuration to ensure src is in python path"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
