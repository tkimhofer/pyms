import pandas as pd
import numpy as np
import datetime as dt
from typing import Union
from pathlib import Path
import json


def write_cache(cls, cache_dir: Union[str, Path]):
    cache_dir = Path(cache_dir)

    points = pd.read_parquet(cache_dir / "datapoints.parquet")
    scans = pd.read_parquet(cache_dir / "scaninfo.parquet")

    with open(cache_dir / "manifest.json") as f:
        manifest = json.load(f)

    xrawd = {}
    dfd = {}

    for mode, g in points.groupby("ms_mode", sort=False):
        X = np.vstack([
            g["scan_id_ori"].to_numpy(),
            g["mz"].to_numpy(),
            g["intensity"].to_numpy(),
            g["rt"].to_numpy(),
            g["scan_id_norm"].to_numpy(),
        ])
        xrawd[mode] = X

    for mode, g in scans.groupby("ms_mode", sort=False):
        dfd[mode] = g.drop(columns=["ms_mode"]).reset_index(drop=True)

    return cls(
        dpath=str(cache_dir),
        fname=manifest["source_file"],
        mslevel=manifest["mslevel"],
        ms0string=manifest["ms0string"],
        ms1string=manifest.get("ms1string"),
        xrawd=xrawd,
        dfd=dfd,
        summary=False,
        import_params=manifest.get("import_params", {}),
    )


def read_cache(exp, cache_dir: Union[str, Path]):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    points = exp.xrawd_to_points(exp.xrawd)

    scans = exp.dfd_to_scans(exp.dfd)

    points.to_parquet(cache_dir / "datapoints.parquet", index=False)
    scans.to_parquet(cache_dir / "scaninfo.parquet", index=False)

    manifest = {
        "format": "msmate-cache-v1",
        "source_file": str(exp.fname),
        "mslevel": exp.mslevel,
        "ms0string": exp.ms0string,
        "ms1string": exp.ms1string,
        "import_params": exp.import_params,
        "stored_at": str(dt.datetime.now())
    }

    with open(cache_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


