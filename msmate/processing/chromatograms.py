import numpy as np
from msmate.core.types import ScanWindow, DBSCANParams, IDX_NOISE, IDX_MZ, IDX_INT, IDX_ST, IDX_SCAN_ORI, IDX_SCAN_NORM


# @log
def _d_ppm(mz: float, ppm: float):
    d = (ppm * mz) / 1e6
    return mz - (d / 2), mz + (d / 2)

def _xic(exp, mz: float, ppm: float, rt_min: float = None, rt_max: float = None):
    mz_min, mz_max = _d_ppm(mz, ppm)

    idx_mz = np.where((exp.xrawd[exp.ms0string][IDX_MZ] >= mz_min) & (exp.xrawd[exp.ms0string][IDX_MZ] <= mz_max))[0]
    if len(idx_mz) == 0:
        raise ValueError('mz range not found')

    X_mz = exp.xrawd[exp.ms0string][..., idx_mz]

    sid = np.array(np.unique(X_mz[0]).astype(int))
    xic = np.zeros(int(np.max(exp.xrawd[exp.ms0string][0]) + 1))
    for i in sid:
        xic[i - 1] = np.sum(X_mz[np.where(X_mz[:, 0] == i), 2])
    stime = np.sort(np.unique(exp.xrawd[exp.ms0string][3]))

    if (~isinstance(rt_min, type(None)) | ~isinstance(rt_max, type(None))):
        idx_rt = np.where((stime >= rt_min) & (stime <= rt_max))[0]
        if len(idx_rt) == 0:
            raise ValueError('rt range not found')
        stime = stime[idx_rt]
        xic = xic[idx_rt]

    return (stime, xic)
