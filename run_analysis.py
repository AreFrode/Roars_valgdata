#!/usr/bin/env python3
"""
Command-line script to run election prediction analysis.
"""

from src.main import main
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


if __name__ == "__main__":
    main()
