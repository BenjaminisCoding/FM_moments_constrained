#!/usr/bin/env python3
"""Prepare raw study data for run_paper.py (see each dataset's --help).

Install: python -m pip install -e '.[observations,single-cell]'
Examples:
  python scripts/preprocess_data.py semrau --raw datasets/raw/semrau --download
  python scripts/preprocess_data.py aerosol --raw datasets/raw/aerosol --download
  python scripts/preprocess_data.py multi --raw datasets/raw/multi --download
  python scripts/run_paper.py --data-root datasets/prepared train --benchmark semrau --method lin --out outputs/semrau

Existing dataset outputs are never overwritten. Each command registers its
files in the output root's manifest. Downloads retrieve only missing,
checksum-verified source files. Multi uses the public Mendeley H5AD with PCA100 and
uses the bundled frozen classifiers; it does not reconstruct PCA from counts.
"""

import argparse
import importlib
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from threadpoolctl import threadpool_limits

from cfm_project.preprocessing.common import register


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset", choices=["semrau", "aerosol", "multi"])
    parser.add_argument(
        "--raw",
        required=True,
        type=Path,
        help="Raw input directory (or Multi H5AD file)",
    )
    parser.add_argument(
        "--out",
        default=Path("datasets/prepared"),
        type=Path,
        help="Prepared benchmark root",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing source files and verify SHA256",
    )
    args = parser.parse_args()
    destination = args.out / args.dataset
    if destination.exists():
        parser.error(f"{destination} already exists; choose a new --out directory")
    args.out.mkdir(parents=True, exist_ok=True)
    module = importlib.import_module(f"cfm_project.preprocessing.{args.dataset}")
    # Publish only complete preparations, so failed runs can be retried safely.
    with tempfile.TemporaryDirectory(prefix=".preparing-", dir=args.out) as temporary:
        staging = Path(temporary) / args.dataset
        staging.mkdir()
        with threadpool_limits(2):
            if args.dataset == "multi":
                import torch

                torch.set_num_threads(2)
            module.prepare(args.raw, staging, args.download)
        staging.rename(destination)
    register(args.out, destination)
    print(f"Prepared {destination}; use run_paper.py --data-root {args.out}")


if __name__ == "__main__":
    main()
