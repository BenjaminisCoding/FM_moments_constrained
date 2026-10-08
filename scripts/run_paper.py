#!/usr/bin/env python3
"""Run the paper's fixed experiments; use --help for preparation and training."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cfm_project.benchmarks.cli import main

if __name__ == '__main__':
    main()
