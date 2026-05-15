
from typeguard import typechecked
# from pathlib import Path
import pandas as pd
import numpy as np
from typing import Union




from msmate.core.types import ScanWindow, DBSCANParams, QCParams, IDX_MZ, IDX_INT, IDX_ST, IDX_SCAN_ORI, IDX_SCAN_NORM
from msmate.core.cache import read_cache, write_cache
from msmate.processing.feature_detection import FeatureDetector
from msmate.io.mz import ReadM
from msmate.viz.plots import PlottingAccessor


@typechecked
class MsExperiment():
    """ Class representing single experiment XC-MS data.
        Methods include read-in, feature detection and visualisation of chromatogram,
        ms scan and chromatographic vs ms dimension.
    """
    @classmethod
    def from_mzfile(cls, fpath: str, scan_window:ScanWindow, mslev:str='1'):
        # cls.__name__ = 'mzml or mzxml from file import'
        da = ReadM(fpath, scan_window, mslev)
        return cls(dpath=da.fpath, fname=da.fpath, mslevel=da.mslevel, ms0string=da.ms0string, ms1string=da.ms1string,
                   xrawd=da.xrawd, dfd=da.dfd, summary=False, import_params=da.import_params,)

    @classmethod
    def from_readm(cls, da: ReadM):
        # cls.__name__ = 'data import from msmate ReadM'
        return cls(dpath=da.fpath, fname=da.fpath, mslevel=da.mslevel, ms0string=da.ms0string, ms1string=da.ms1string,
                   xrawd=da.xrawd, dfd=da.dfd, summary=False, import_params=da.import_params,)

    @property
    def plot(self):
        return PlottingAccessor(self)
    def __init__(self,
                 dpath:str, fname:str,
                 mslevel:str, ms0string:str,
                 ms1string:Union[str, None],
                 xrawd:dict,
                 dfd:dict,
                 summary:bool,
                 import_params: Union[dict, None] = None,
                 ):
        self.mslevel = mslevel
        self.dpath = dpath
        self.fname = fname

        self.ms0string = ms0string
        self.ms1string = ms1string
        self.xrawd = xrawd # usage: self.xrawd[self.ms0string]
        self.dfd = dfd # usage: self.dfd[self.ms0string], scantype eg., '0_2_2_5_20.0'
        self.summary = summary
        self.import_params = import_params or {}

        df = self.dfd[self.ms0string]

        # renaming the minute time unit as standardise to seconds
        idc = [i for i, c in enumerate(df.columns) if c == "time_unit"]

        if len(idc) > 1:
            cols = list(df.columns)

            for i in idc:
                col_values = df.iloc[:, i]

                # check if this column contains 'minute'
                if (col_values == "minute").any():
                    cols[i] = "time_unit_ori"
                    break  # only rename one

            df.columns = cols
            self.dfd[self.ms0string] = df


        if 'tic' not in self.dfd[self.ms0string].columns:
            # only if not present
            self.append_scan_summaries()

    def append_scan_summaries(self):
        arr = self.xrawd[self.ms0string]

        sid = arr[IDX_SCAN_ORI]
        mz = arr[IDX_MZ]
        inten = arr[IDX_INT]

        # group scan ids
        unique_sid, inv = np.unique(sid, return_inverse=True)
        n = len(unique_sid)

        # max intensity per scan
        max_int = np.full(n, -np.inf)
        np.maximum.at(max_int, inv, inten)

        # sum intensity per scan
        sum_int = np.zeros(n, dtype=float)
        np.add.at(sum_int, inv, inten)

        # m/z at max intensity per scan
        # sort by scan id, then intensity descending
        order = np.lexsort((-inten, sid))
        sid_sorted = sid[order]

        first_idx = np.r_[True, sid_sorted[1:] != sid_sorted[:-1]]
        top_rows = order[first_idx]

        # top_sid = sid[top_rows]
        top_mz = mz[top_rows]
        # top_int = inten[top_rows]

        scan_summary = pd.DataFrame({
            "index": unique_sid.astype(int),
            "tic": sum_int,
            "bpi": max_int,
            "bp_mz": top_mz,
        })

        self.dfd[self.ms0string]["index"] = (
            self.dfd[self.ms0string]["index"].astype(int)
        )

        self.dfd[self.ms0string] = (
            self.dfd[self.ms0string]
            .merge(scan_summary, on="index", how="left")
        )

    @staticmethod
    def xrawd_to_points(xrawd: dict) -> pd.DataFrame:
        dfs = []

        for mode, X in xrawd.items():
            dfs.append(pd.DataFrame({
                "ms_mode": mode,  # "1P", "1N"
                "scan_id_ori": X[IDX_SCAN_ORI].astype("int32"),
                "mz": X[IDX_MZ].astype("float32"),
                "intensity": X[IDX_INT].astype("float32"),
                "rt": X[IDX_ST].astype("float32"),
                "scan_id_norm": X[IDX_SCAN_NORM].astype("int32"),
            }))

        return pd.concat(dfs, ignore_index=True)

    def to_cache(self, cache_dir):
        write_cache(self, cache_dir)

    @classmethod
    def from_cache(cls, cache_dir):
        data = read_cache(cache_dir)
        return cls(**data)

    @staticmethod
    def dfd_to_scans(dfd: dict) -> pd.DataFrame:
        dfs = []

        for mode, df in dfd.items():
            tmp = df.copy()
            tmp["ms_mode"] = mode
            dfs.append(tmp)

        return pd.concat(dfs, ignore_index=True)


    def init_detector(self):
        self.detector = FeatureDetector(self)

    def detect_features(self, qc_pars:QCParams=QCParams(), **kwargs):
        self.Xff, self.cluster, self.cluster_description = self.detect_features_stateless(qc_pars, **kwargs)

    def detect_features_stateless(self, qc_pars:QCParams=QCParams(), **kwargs):
        detector = FeatureDetector(self)
        Xff, cluster = detector.dbscan_detect(**kwargs)
        desc = detector.cluster_summary(cluster, Xff, qc_pars, exhaustive=False)
        return Xff, cluster, desc

    def get_features(self, qc_pars: QCParams = QCParams(), **kwargs):
        detector = FeatureDetector(self)
        Xff, cluster = detector.dbscan_detect(**kwargs)
        desc = detector.cluster_summary(cluster, Xff, qc_pars, exhaustive=False)
        return desc


