#!/usr/bin/env python3
"""Run the fixed scarce-sample, redundant-mean and optical-noise studies."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cfm_project.benchmarks.ablations import main
if __name__ == '__main__': main()
