import numpy as np
import itertools
import pandas as pd
import re
from typing import Union

class IsotopeFinder:
    def __init__(self, features: dict):
        self.features = features
        self.df = self._to_frame(features)

    @staticmethod
    def _to_frame(features: dict) -> pd.DataFrame:
        df = pd.DataFrame.from_dict(features, orient="index")
        df.index.name = "fid"

        required = ["apex_mz", "apex_st", "height", "st_min", "st_max"]
        missing = [c for c in required if c not in df.columns]

        if missing:
            raise ValueError(f"Missing required feature fields: {missing}")

        return df.sort_values("height", ascending=False)

    def find_patterns(
        self,
        mz_tol: float = 0.005,
        rt_tol: float = 1.0,
        abundance_lb: float = 0.005,
        abundance_ub: float = 0.8,
        max_iso: int = 3,
        min_qc_score: Union[float, None] = 0.2,
        charge: int = 1,
        intensity_col: str = "height",
    ):
        df = self.df.copy()

        if min_qc_score is not None and "qc_score" in df.columns:
            df = df[df["qc_score"] >= min_qc_score]

        active = set(df.index)
        patterns = {}
        isotope_spacing = 1.003355 / abs(charge)

        pat_id = 0

        for fid in df.index:
            if fid not in active:
                continue

            base = df.loc[fid]
            pattern = [fid]
            last_fid = fid

            for iso_n in range(1, max_iso + 1):

                last = df.loc[last_fid]

                target_mz = last["apex_mz"] + isotope_spacing
                base_int = base[intensity_col]

                candidates = df.loc[list(active)]

                rt_match = np.abs(candidates["apex_st"] - base["apex_st"]) <= rt_tol
                mz_match = np.abs(candidates["apex_mz"] - target_mz) <= mz_tol

                lower = base_int * abundance_lb
                upper = base_int * abundance_ub

                int_match = (
                    (candidates[intensity_col] >= lower) &
                    (candidates[intensity_col] <= upper)
                )

                # Optional: require RT-window overlap
                rt_overlap = (
                    (candidates["st_min"] <= base["st_max"]) &
                    (candidates["st_max"] >= base["st_min"])
                )

                hit = candidates[rt_match & mz_match & int_match & rt_overlap]

                if hit.empty:
                    break

                # Pick best candidate: smallest mz error, then best RT match
                hit = hit.assign(
                    mz_err=np.abs(hit["apex_mz"] - target_mz),
                    rt_err=np.abs(hit["apex_st"] - base["apex_st"]),
                ).sort_values(["mz_err", "rt_err"], ascending=True)

                next_fid = hit.index[0]
                pattern.append(next_fid)
                last_fid = next_fid

            if len(pattern) > 1:
                patterns[pat_id] = pattern

                for x in pattern:
                    active.discard(x)

                pat_id += 1
            else:
                active.discard(fid)

        return self._annotate_patterns(patterns)

    def _annotate_patterns(self, patterns: dict):
        df = self.df.copy()
        df["iso_pattern_id"] = None
        df["iso_label"] = None
        df["iso_n"] = np.nan

        for pat_id, fids in patterns.items():
            for iso_n, fid in enumerate(fids):
                df.loc[fid, "iso_pattern_id"] = pat_id
                df.loc[fid, "iso_label"] = f"M+{iso_n}"
                df.loc[fid, "iso_n"] = iso_n

        self.patterns = patterns
        self.result = df
        return df

# finder = IsotopeFinder(features=feat, feature_table=l3Df)
# result = finder.find(mz_tol=0.01, abundance_ratio=0.2)
#
# class IsotopeFinder:
#     def __init__(self, features, feature_table):
#         self.features = features
#         self.feat = features
#         self.l3Df = feature_table.copy()
#     def find_patterns(self, mz_tol=0.01, a_lb=0.2):  # feat, l3Df,
#         # self.feat =feat
#         self.l3Df = self.l3Df.sort_values('sum_intensity', ascending=False)
#         self.mz_tol = mz_tol
#         self.a_lb = a_lb
#         self.fset = dict.fromkeys(self.l3Df.index.values)
#         self.ipat = {}
#         isoID = np.repeat(None, self.l3Df.shape[0])
#         ionID = np.repeat(None, self.l3Df.shape[0])
#
#         # for each detected feature in fset, find isotopologues
#         icounter = 0
#         while len(self.fset) > 0:
#             fid = list(self.fset.keys())[0]
#
#             try:
#                 st_prop = self._fwhmBound_iso(fid, rtDelta=1)
#
#             except (KeyError, ValueError, IndexError) as e:
#                 # logger.debug("Skipping feature %s: %s", fid, e)
#                 self.fset.pop(fid, None)
#                 continue
#
#             if st_prop is None:
#                 self.fset.pop(fid)
#                 continue
#
#             isto = self.getIP(m=None, df=st_prop, iid=fid, mz_tol=0.1, a_lb=0.2)
#
#             if len(isto.fid) > 1:
#                 self.ipat[fid] = isto
#                 iids = [x.id for k, x in isto.fid.items()]
#                 ils = np.where(self.l3Df.index.isin(iids))[0]
#                 isoID[ils] = icounter
#                 ionID[ils] = [f'{icounter}_M{i}' for i in range(len(iids))]
#                 icounter += 1
#                 [self.fset.pop(x) for x in iids]
#             else:
#                 self.fset.pop(fid)
#
#         self.l3Df['iPat'] = ionID
#         self.l3Df = self.l3Df.sort_values('iPat')
#
#     def _extend_pattern(self, m, df, iid, mz_tol=0.1, a_lb=0.2):
#         ### a_lb => isotope count diff (eg 0.2 -> 20% of main signal)
#
#         # df are feature proposals based on st
#         if m is None:
#             # print('none')
#             m = IsoPat(df.loc[iid])
#             return self.getIP(m, df, iid)
#
#         m0_mz = m.fid[f'{m.c}']['mzMaxI']
#         m0_a = m.fid[f'{m.c}']['smbl']
#
#         dmz = df['mzMaxI'] - m0_mz
#         imz = (dmz > (1 - mz_tol)) & (dmz < (1 + mz_tol))
#
#         ia = df['smbl'] < (m0_a * a_lb)
#
#         idx = np.where((imz & ia))[0]
#
#         if len(idx) > 0:
#             m.add(df.iloc[idx[0]])
#             return self._extend_pattern(m, df, df.index[idx[0]], mz_tol, a_lb)
#
#         return m
#
#     def rt_fwhm_window(self, i, rtDelta=1 - 0.4):
#         # import matplotlib.pyplot as plt
#         intens = self.feat[i]['fdata']['I_sm_bline']
#         # intens1 = self.feat[i]['fdata']['I_raw']
#         st = self.feat[i]['fdata']['st']
#         # plt.plot(st, intens)
#         # plt.plot(st, intens1)
#         idxImax = np.argmax(intens)
#         ifwhm = intens[idxImax] / 2
#         if (idxImax <= 2) or (idxImax > (len(intens) - 2)):
#             return None
#         st_imax = st[idxImax]
#         ll = np.max(st[(intens < ifwhm) & (st <= st_imax)])
#         ul = np.min(st[(intens < ifwhm) & (st >= st_imax)])
#         ll1 = st_imax - ((ul - ll) * rtDelta) / 2
#         ul1 = st_imax + ((ul - ll) * rtDelta) / 2
#         # print(f'st low: {ll1} and st high: {ul1}')
#         sub = self.l3Df[
#             (self.l3Df['rtMaxI'] <= ul1) & (self.l3Df['rtMaxI'] >= ll1) & self.l3Df.index.isin(self.fset.keys())]
#         if sub.shape[0] == 0:
#             return None
#         else:
#             return sub
#
#     def _fwhmBound(self, i, rtDelta=1 - 0.4):
#         # import matplotlib.pyplot as plt
#         intens = self.feat[i]['fdata']['I_sm_bline']
#         # intens1 = self.feat[i]['fdata']['I_raw']
#         st = self.feat[i]['fdata']['st']
#         # plt.plot(st, intens)
#         # plt.plot(st, intens1)
#         idxImax = np.argmax(intens)
#         ifwhm = intens[idxImax] / 2
#         if (idxImax <= 2) or (idxImax > (len(intens) - 2)):
#             return None
#         st_imax = st[idxImax]
#         ll = np.max(st[(intens < ifwhm) & (st <= st_imax)])
#         ul = np.min(st[(intens < ifwhm) & (st >= st_imax)])
#         ll1 = st_imax - ((ul - ll) * rtDelta) / 2
#         ul1 = st_imax + ((ul - ll) * rtDelta) / 2
#
#         return (ll1, ul1)
#         #
#         # # print(f'st low: {ll1} and st high: {ul1}')
#         # sub = self.l3Df[(self.l3Df['rtMaxI'] <= ul1) & (self.l3Df['rtMaxI'] >= ll1)]
#         # if sub.shape[0] == 0:
#         #     return None
#         # else:
#         #     return sub
#
