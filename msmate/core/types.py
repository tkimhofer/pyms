from dataclasses import dataclass


IDX_SCAN_ORI = 0
IDX_MZ = 1
IDX_INT = 2
IDX_ST = 3
IDX_SCAN_NORM = 4
IDX_NOISE = 5
# IDX_MZ_SCALED = 6
IDX_CLUSTER = 6

@dataclass
class ScanWindow:
    mz_min: float = 50
    mz_max: float = 600
    st_min: float = 60
    st_max: float = 600

@dataclass
class DBSCANParams:
    # defining scaling factors for mz and st dimension to achieve isotropy (required for eps)
    mz_sig: float = 5.,
    st_sig: float = 1.,

    # defining core sample crit: `min_samples` exist in distance eps (density crit)
    min_samples: int = 5  # scan frequency: 5 Hz
    eps: float = 1

    # definition of noise threshold using quantile probability
    q_noise: float = 0.99

@dataclass
class QCParams:
    min_points: int = 5
    st_len_min: int = 5
    sid_gap: float = 0.2 # this is mean value, ie allowing small sid gaps
    ppm: float = 30.0
    raggedness: float = 0.15
    non_neg: float = 0.8
    sino: float = 2.0
