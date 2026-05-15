

from msmate.core.experiment import MsExperiment
from msmate.core.types import ScanWindow, DBSCANParams, QCParams
from msmate.isotopes.grouping import IsotopeFinder
from msmate.io.helpers_xml import inspect_msfile
from msmate.processing.parameter_optimisation import  score_runs, score_stability_fast

# t0 = time.perf_counter()
# path='mz_files/Beer_multibeers_1_fullscan1.mzXML'
path = 'mz_files/Urine_HILIC_ESIpos_msLevel1.mzXML'
# path = 'mz_files/VGF138_A01_neg.mzML'

# inspect mz file for experiment details
inspect_msfile(path)

# define region of interest and import MS1 data
swin = ScanWindow(mz_min=30, mz_max=1000, st_min=30, st_max=500)
exp = MsExperiment.from_mzfile(path, scan_window=swin)

# detect features using different parameter sets
runs, features = score_runs(exp, swin)

# calculate feature consensus and provide confidence scores
consensus, features = score_stability_fast(features, runs)

# viz features
fig = exp.plot.consensus_feature(
    consensus_id=consensus.iloc[1001]["consensus_id"],
    consensus=consensus,
    features=features,
)