"""et-fus-vbm: VBM of MRgFUS thalamotomy for essential tremor from deepmriprep outputs."""
from __future__ import annotations
from pathlib import Path
import yaml

__version__ = "0.1.0"


def load_config(path: str | Path) -> dict:
    """Load cohort.yaml and resolve relative paths against the repo root.

    The repo root is taken as the parent of the config file's directory
    (config/cohort.yaml -> repo root). Absolute paths are left untouched.
    """
    path = Path(path).resolve()
    repo_root = path.parent.parent
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_repo_root"] = repo_root

    def _resolve(p):
        if p is None:
            return None
        p = Path(p)
        return p if p.is_absolute() else (repo_root / p)

    for key in ("data_root", "manifest_csv", "derivatives", "upstream_root",
                "mni_mask", "ctrl_dist", "roi_dir", "demographics_csv",
                "cerebellar_csv", "clinical_csv"):
        if key in cfg:
            cfg[key] = _resolve(cfg[key])
    return cfg
