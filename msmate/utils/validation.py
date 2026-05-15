from pathlib import Path
from typing import Union
import numpy as np

from msmate.core.types import ScanWindow, DBSCANParams, QCParams


def ensure_path_exists(path: Union[str, Path]) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    return path


def ensure_supported_mz_file(path: Union[str, Path]) -> Path:
    path = ensure_path_exists(path)
    suffix = path.suffix.lower()

    if suffix not in {".mzml", ".mzxml"}:
        raise ValueError(f"Unsupported MS file format: {suffix}")

    return path


def validate_scan_window(window: ScanWindow) -> ScanWindow:
    if window.mz_min >= window.mz_max:
        raise ValueError("mz_min must be smaller than mz_max.")

    if window.st_min >= window.st_max:
        raise ValueError("st_min must be smaller than st_max.")

    if window.mz_min < 0:
        raise ValueError("mz_min must be non-negative.")

    if window.st_min < 0:
        raise ValueError("st_min must be non-negative.")

    return window


def validate_dbscan_params(params: DBSCANParams) -> DBSCANParams:
    if params.eps <= 0:
        raise ValueError("DBSCAN eps must be > 0.")

    if params.min_samples < 1:
        raise ValueError("DBSCAN min_samples must be >= 1.")

    if params.mz_sig <= 0:
        raise ValueError("mz_sig must be > 0.")

    if params.st_sig <= 0:
        raise ValueError("st_sig must be > 0.")

    if not 0 < params.q_noise < 1:
        raise ValueError("q_noise must be between 0 and 1.")

    return params


def validate_qc_params(params: QCParams) -> QCParams:
    if params.min_points < 1:
        raise ValueError("min_points must be >= 1.")

    if params.st_len_min < 1:
        raise ValueError("st_len_min must be >= 1.")

    if params.ppm <= 0:
        raise ValueError("ppm must be > 0.")

    if not 0 <= params.sid_gap <= 1:
        raise ValueError("sid_gap must be in [0, 1].")

    if not 0 <= params.raggedness <= 1:
        raise ValueError("raggedness must be in [0, 1].")

    if not 0 <= params.non_neg <= 1:
        raise ValueError("non_neg must be in [0, 1].")

    if params.sino < 0:
        raise ValueError("sino must be >= 0.")

    return params


def validate_xraw_matrix(X: np.ndarray, min_rows: int = 5) -> np.ndarray:
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy ndarray.")

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    if X.shape[0] < min_rows:
        raise ValueError(f"X must have at least {min_rows} rows.")

    if X.shape[1] == 0:
        raise ValueError("X contains no datapoints.")

    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values.")

    return X


def validate_quantile(q: float, name: str = "q") -> float:
    if not 0 < q < 1:
        raise ValueError(f"{name} must be between 0 and 1.")
    return q