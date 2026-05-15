import numpy as np
from msmate.core.types import ScanWindow, DBSCANParams, IDX_NOISE, IDX_MZ, IDX_INT, IDX_ST, IDX_SCAN_ORI, IDX_SCAN_NORM


def _window_mz_rt(Xr: np.ndarray, selection: ScanWindow = None):
    """2D filter function using retention/scan time and mz dimension.
    Input shape Xr is rows - mz, st, etc, and columns is scan id / rt
    """
    mask = np.ones(Xr.shape[1], dtype=bool)

    if selection.mz_min is not None:
        mask &= Xr[IDX_MZ] >= selection.mz_min

    if selection.mz_max is not None:
        mask &= Xr[IDX_MZ] <= selection.mz_max

    if selection.st_min is not None:
        mask &= Xr[IDX_ST] > selection.st_min

    if selection.st_max is not None:
        mask &= Xr[IDX_ST] < selection.st_max

    return Xr[..., mask]
