from __future__ import annotations

import os
from pathlib import Path
import sys

import hydra
from omegaconf import DictConfig

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
CACHE_DIR = ROOT / ".cache"
MPL_CONFIG = CACHE_DIR / "matplotlib"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MPL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG))

from cfm_project.pipeline import run_pipeline


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    result = run_pipeline(cfg, output_dir=Path.cwd())
    print(result)


if __name__ == "__main__":
    main()
