
import numpy as np
from sklearn.cluster import DBSCAN
from typing import Union
from collections import defaultdict

from msmate.core.types import ScanWindow, DBSCANParams, QCParams, IDX_NOISE, IDX_MZ, IDX_INT, IDX_ST, IDX_SCAN_ORI, IDX_SCAN_NORM
from msmate.processing.feature_qc import _cluster_summary, _per_cluster_summary
from msmate.utils.filters import _window_mz_rt

class FeatureDetector:
    """ Base class defining feature detection functions operating on 2D MS data.

    Note that this class is designed for use with `MsExp`.
    """

    def __init__(self, experiment):
        self.exp = experiment

    def dbscan_detect(
            self,
            # X:np.ndarray,
            dbs_pars:DBSCANParams,
            scan_window: Union[None, ScanWindow] = None,
           ):

        X = self.exp.xrawd[self.exp.ms0string]

        if scan_window is not None:
            X = _window_mz_rt(X, selection=scan_window)

        # dbscan feature detection after anisotopic scaling
        Xf = self.label_noise(X, q_noise=dbs_pars.q_noise, qcm_local=True)
        idc = np.where(Xf[IDX_NOISE] == 0)[0]

        # scaling to establish anisotopy, then use euclidean dist (faster than scaling in custom distance fun)
        st = Xf[IDX_SCAN_NORM, idc] / dbs_pars.st_sig
        mz = Xf[IDX_MZ, idc]

        # move mz into log space as this represents ppm accuracy better
        mz_ppm = np.log(mz) * 1e6 / dbs_pars.mz_sig

        # scale input variables and use euclid.d as - this is faster than scaling in custom distance fun
        X = np.column_stack([
            st,
            mz_ppm
        ]).astype(np.float32, copy=False)

        dbs = DBSCAN(
            eps=dbs_pars.eps,
            min_samples=dbs_pars.min_samples,
            metric="euclidean",
            algorithm="kd_tree",
            n_jobs=1 # single process here, run in parallelisation
        )

        fits = dbs.fit(X)

        cluster_add = np.full(Xf.shape[1], -1, dtype=int)
        cluster_add[idc] = fits.labels_  # cl_labels
        Xff = np.r_[Xf, cluster_add[np.newaxis, ...]]

        clusters = {
            cid: idxs
            for cid, idxs in self.iter_clusters(Xff[-1].astype(int))
        }

        # clusters = {}
        # for i, cid in enumerate(Xff[-1]):
        #     clusters.setdefault(cid, []).append(i)

        return Xff, clusters

    @staticmethod
    def iter_clusters(labels):
        groups = defaultdict(list)

        for idx, cid in enumerate(labels):
            if cid >= 0:
                groups[cid].append(idx)

        for cid, idxs in groups.items():
            yield cid, np.asarray(idxs)

    def dbscan_detect_qc(
            self,
            dbs_pars:DBSCANParams,
            qc_par: QCParams,
            scan_window: Union[None, ScanWindow] = None,
           ):

        X = self.exp.xrawd[self.exp.ms0string]

        if scan_window is not None:
            X = _window_mz_rt(X, selection=scan_window)

        noise_thres = np.quantile(X[IDX_INT], q=dbs_pars.q_noise)
        idc_signal = np.where(X[IDX_INT] > noise_thres)[0]

        # clustering taking place in st/mz dim
        st = X[IDX_SCAN_NORM, idc_signal]

        # scaling to establish isotopy, then use euclidean dist (faster than scaling in custom distance fun)
        # move mz into log space as this represents ppm accuracy better
        mz = X[IDX_MZ, idc_signal]
        mz_ppm = (np.log(mz) * 1e6 / dbs_pars.mz_sig) # mz_sig...-> ppm like softness

        # scale input variables and use euclid.d as - this is faster than scaling in custom distance fun
        Xsc = np.column_stack([
            st / dbs_pars.st_sig,
            mz_ppm
        ]).astype(np.float32, copy=False)

        dbs = DBSCAN(
            eps=dbs_pars.eps,
            min_samples=dbs_pars.min_samples,
            metric="euclidean",
            algorithm="kd_tree",
            n_jobs=1 # single process here, run in parallelisation
        )

        fits = dbs.fit(Xsc)
        cl = self.iter_clusters(fits.labels_)

        intens = X[IDX_INT, idc_signal]
        st_sec = X[IDX_ST, idc_signal]

        feats = {}
        for cid, cl_idc in cl:

            summary = _per_cluster_summary(
                st=st_sec[cl_idc],
                intens=intens[cl_idc],
                mz=mz[cl_idc],
                sid=st[cl_idc],
                qc_par=qc_par
            )

            if summary:
                feats[cid] = summary

        return feats



    @staticmethod
    def cluster_summary(cluster:dict, Xf:np.ndarray, qc_par:DBSCANParams, exhaustive:bool):
        return _cluster_summary(cluster, Xf, qc_par, exhaustive)

    @staticmethod
    def label_noise(Xf, q_noise, qcm_local, ):

        if qcm_local:
            noise_thres = np.quantile(Xf[IDX_INT], q=q_noise)
        else:
            raise ValueError('not defined')

        # filter noise data points
        idx_signal = np.where(Xf[IDX_INT] > noise_thres)[0]
        # print(f'Number of dp: {len(idx_signal)}')

        noise = np.ones(Xf.shape[1])
        noise[idx_signal] = 0
        return np.r_[Xf, noise[np.newaxis, ...]]
