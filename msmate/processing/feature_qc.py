import numpy as np
from scipy.signal import find_peaks
from msmate.core.types import  QCParams, IDX_NOISE, IDX_MZ, IDX_INT, IDX_ST, IDX_SCAN_ORI, IDX_SCAN_NORM

def _cluster_checkpoint1_pass_fail(cl_idx: list, Xf: np.ndarray, qc_par: QCParams):  # Xf, qc_par
    """First cluster checkpoint: early decision fail or pass.
     Continue with more expensive summary stats with passing / reduced subset
     """

    n = len(cl_idx)
    if n < qc_par.min_points:
        return False, {'crit': 'min_points', 'n': n}

    st = Xf[IDX_ST, cl_idx]
    intens = Xf[IDX_INT, cl_idx]
    mz = Xf[IDX_MZ, cl_idx]
    sid = Xf[IDX_SCAN_NORM, cl_idx]

    idx_maxI = np.argmax(intens)
    if idx_maxI < 2 or idx_maxI > n - 2:
        return False, {"crit": "apex_bordered", "n": n, "icent": idx_maxI}

    # ppm is huge due to low intensity fringe points
    # using from intensity-scaled cluster points (de-emph fringe points)
    ### note: there can also be multiple signals in feature boundary -> high ppm inbetween

    sI = intens.sum()
    mz_w = np.dot(mz, intens) / sI
    ppm = 1e6 * (mz - mz_w) / mz_w

    ppm_wad = np.dot(np.abs(ppm), intens) / sI
    # ppm_wrms = np.sqrt(np.dot(ppm * ppm, intens) / sI)

    # mz_maxI = mz[idx_maxI]
    # ppm = (mz.max() - mz.min()) / mz_maxI * 1e6
    if ppm_wad > qc_par.ppm:
        return False, {"crit": "mz_ppm", "ppm": ppm_wad, "n": n, "icent": idx_maxI}

    sid_diff = np.diff(sid)
    if np.any(sid_diff <= 0):
        return False, {"crit": "scan_ids_not_strictly_increasing",  "ppm": ppm_wad, "n": n, "icent": idx_maxI}


    # again fringe points (ie tails) misbehave, cheking for gaps in singal center
    # define core quickly
    # if core_mode == "median":
    #     thr = np.median(inten)
    #     keep = inten >= thr
    # elif core_mode == "q25":
    thr = np.quantile(intens, 0.5)
    keep = intens >= thr
    # else:
    #     raise ValueError("unknown core_mode")
    sid_diff = np.diff(sid[keep])
    sid_gap = np.mean(sid_diff > 1)
    if sid_gap > qc_par.sid_gap:
        return False, {"crit": "scans_gaps", "sId_gap": sid_gap,  "ppm": ppm_wad, "n": n, "icent": idx_maxI}

    return True, {
        "n": n,
        "icent": idx_maxI,
        "ppm": ppm_wad,
        "st_span": st.max() - st.min(),
        "sId_gap": sid_gap,
        # "st_span": sid
    }


def _cluster_checkpoint1_pass_fail1(st, intens, mz, sid, qc_par: QCParams):
    """First cluster checkpoint: early decision fail or pass.
     Continue with more expensive summary stats with passing / reduced subset
     """

    n = len(st)
    if n < qc_par.min_points:
        return False, {'crit': 'min_points', 'n': n}

    idx_maxI = np.argmax(intens)
    if idx_maxI < 2 or idx_maxI > n - 2:
        return False, {"crit": "apex_bordered", "n": n, "icent": idx_maxI}

    # ppm is huge due to low intensity fringe points
    # using from intensity-scaled cluster points (de-emph fringe points)
    ### note: there can also be multiple signals in feature boundary -> high ppm inbetween
    sI = intens.sum()
    mz_w = np.dot(mz, intens) / sI
    ppm = 1e6 * (mz - mz_w) / mz_w

    ppm_wad = np.dot(np.abs(ppm), intens) / sI

    if ppm_wad > qc_par.ppm:
        return False, {"crit": "mz_ppm", "ppm": ppm_wad, "n": n, "icent": idx_maxI}

    sid_diff = np.diff(sid)
    if np.any(sid_diff <= 0):
        return False, {"crit": "scan_ids_not_strictly_increasing",  "ppm": ppm_wad, "n": n, "icent": idx_maxI}

    thr = np.quantile(intens, 0.5)
    keep = intens >= thr

    sid_diff = np.diff(sid[keep])
    sid_gap = np.mean(sid_diff > 1)
    if sid_gap > qc_par.sid_gap:
        return False, {"crit": "scans_gaps", "sId_gap": sid_gap,  "ppm": ppm_wad, "n": n, "icent": idx_maxI}

    return True, {
        "n": n,
        "icent": idx_maxI,
        "ppm": ppm_wad,
        "st_span": st.max() - st.min(),
        "sId_gap": sid_gap,
    }


def _fail(crit, qc, quant, descr, fdata, flev=2):
    return {
        'flev': flev,
        'crit': crit,
        'qc': qc,
        'quant': quant,
        'descr': descr,
        'fdata': fdata,
    }

def _cluster_description(st, mz, intens):
    """Fast LC-MS cluster / peak descriptors.

    Inputs should be 1D arrays belonging to one DBSCAN cluster.
    Returns cheap descriptors useful for peak QC, split detection,
    merge detection and later isotope-pattern checks.

    feature / peak descriptors:
        geometry: rt/mz span, apex position, ppm spread
        continuity: scan-id gaps
        shape: raggedness, peak count, prominence
        quality: s/n, baseline negativity, area

    """
    st = np.asarray(st, dtype=float)
    mz = np.asarray(mz, dtype=float)
    intens = np.asarray(intens, dtype=float)

    if len(st) == 0:
        raise ValueError("Empty cluster.")

    # sort by scan/retention time
    order = np.argsort(st)
    st = st[order]
    mz = mz[order]
    intens = intens[order]

    n = len(st)

    st_min = float(st.min())
    st_max = float(st.max())
    st_span = float(st_max - st_min)

    mz_min = float(mz.min())
    mz_max = float(mz.max())

    height = float(intens.max())
    apex_idx = int(np.argmax(intens))

    apex_st = float(st[apex_idx])
    apex_mz = float(mz[apex_idx])

    sum_intensity = float(intens.sum())

    if n >= 2:
        area = float(np.trapz(intens, st))
    else:
        area = 0.0

    # intensity-weighted m/z descriptor
    if sum_intensity > 0:
        mz_wmean = float(np.dot(mz, intens) / sum_intensity)
        ppm_dev = 1e6 * (mz - mz_wmean) / mz_wmean
        ppm_wad = float(np.dot(np.abs(ppm_dev), intens) / sum_intensity)
    else:
        mz_wmean = float(np.mean(mz))
        ppm_wad = np.nan

    # raw ppm span, useful but sensitive to fringe points
    ppm_span = float((mz_max - mz_min) / max(apex_mz, 1e-12) * 1e6)

    # incomplete / cut peak indicators
    if height > 0:
        edge_left = float(intens[0] / height)
        edge_right = float(intens[-1] / height)
        edge_max = float(max(edge_left, edge_right))
    else:
        edge_left = edge_right = edge_max = np.nan

    apex_at_edge = bool(apex_idx < 2 or apex_idx > n - 3)

    # simple smoothing for shape descriptors
    if n >= 3:
        y = np.convolve(intens, np.array([0.25, 0.5, 0.25]), mode="same")
        y[0] = intens[0]
        y[-1] = intens[-1]
    else:
        y = intens.copy()

    ymax = float(y.max()) if len(y) else 0.0

    # raggedness: violations of rise-before-apex / fall-after-apex
    if n >= 3 and ymax > 0:
        aidx = int(np.argmax(y))
        tol = 0.02 * ymax

        left_viol = np.sum(np.diff(y[:aidx + 1]) < -tol)
        right_viol = np.sum(np.diff(y[aidx:]) > tol)

        raggedness = float((left_viol + right_viol) / max(n - 1, 1))
    else:
        raggedness = np.nan

    # peak count and valley ratio for merged/split features
    n_major_peaks = 0
    min_valley_ratio = np.nan

    if n >= 5 and ymax > 0:
        peaks, props = find_peaks(
            y,
            prominence=0.15 * ymax,
            distance=2,
        )

        n_major_peaks = int(len(peaks))

        if n_major_peaks >= 2:
            valley_ratios = []

            peaks = np.sort(peaks)
            for p1, p2 in zip(peaks[:-1], peaks[1:]):
                lo, hi = sorted((p1, p2))
                valley = float(np.min(y[lo:hi + 1]))
                smaller_peak = float(min(y[p1], y[p2]))
                valley_ratios.append(valley / max(smaller_peak, 1e-12))

            min_valley_ratio = float(np.min(valley_ratios))

    return {
        "n": n,

        "st_min": st_min,
        "st_max": st_max,
        "st_span": st_span,

        "mz_min": mz_min,
        "mz_max": mz_max,
        "mz_wmean": mz_wmean,

        "apex_idx": apex_idx,
        "apex_st": apex_st,
        "apex_mz": apex_mz,
        "apex_at_edge": apex_at_edge,

        "height": height,
        "sum_intensity": sum_intensity,
        "area": area,

        "ppm_span": ppm_span,
        "ppm_wad": ppm_wad,

        "edge_left": edge_left,
        "edge_right": edge_right,
        "edge_max": edge_max,

        "raggedness": raggedness,

        "n_major_peaks": n_major_peaks,
        "min_valley_ratio": min_valley_ratio,
    }


def classify_one(d):
    flags = []

    n = d.get("n", np.nan)
    ppm_wad = d.get("ppm_wad", d.get("ppm", np.nan))
    sid_gap = d.get("sId_gap", d.get("sid_gap", 0))
    edge_max = d.get("edge_max", np.nan)
    raggedness = d.get("raggedness", np.nan)
    n_major_peaks = d.get("n_major_peaks", 0)
    min_valley_ratio = d.get("min_valley_ratio", np.nan)
    apex_at_edge = d.get("apex_at_edge", False)

    if n < 5:
        flags.append("too_few_points")

    if apex_at_edge:
        flags.append("apex_at_edge")

    if ppm_wad > 30:
        flags.append("mz_unstable")

    if sid_gap > 0.2:
        flags.append("scan_gaps")

    if not np.isnan(edge_max):
        if edge_max > 0.5:
            flags.append("incomplete_or_cut_peak")
        elif edge_max > 0.25:
            flags.append("suspicious_edges")

    if not np.isnan(raggedness):
        if raggedness > 0.25:
            flags.append("ragged_shape")
        elif raggedness > 0.15:
            flags.append("slightly_ragged")

    if n_major_peaks >= 2:
        if not np.isnan(min_valley_ratio):
            if min_valley_ratio < 0.5:
                flags.append("split_or_merged_peak")
            elif min_valley_ratio < 0.8:
                flags.append("possible_shoulder_or_partial_merge")
        else:
            flags.append("multiple_peaks")

    hard_bad = {
        "too_few_points",
        "mz_unstable",
        "apex_at_edge",
    }

    poor = {
        "incomplete_or_cut_peak",
        "ragged_shape",
        "split_or_merged_peak",
    }

    warning = {
        "scan_gaps",
        "suspicious_edges",
        "slightly_ragged",
        "possible_shoulder_or_partial_merge",
        "multiple_peaks",
    }

    if not flags:
        label = "good"
    elif any(f in hard_bad for f in flags):
        label = "bad"
    elif any(f in poor for f in flags):
        label = "poor"
    elif any(f in warning for f in flags):
        label = "usable_with_warning"
    else:
        label = "unknown"

    return label, flags

# peak filter
def _classify_feature_quality(features):
    """Classify LC-MS feature descriptors.

    Parameters
    ----------
    features : list[dict] or pandas.DataFrame
        Output from cluster/feature description.

    Returns
    -------
    pandas.DataFrame
        Feature descriptors plus quality_label and quality_flags.
    """
    import pandas as pd

    df = pd.DataFrame(features).copy()

    labels = []
    flags = []

    for row in df.to_dict(orient="records"):
        label, flg = classify_one(row)
        labels.append(label)
        flags.append(flg)

    df["quality_label"] = labels
    df["quality_flags"] = flags
    df["quality_flags_str"] = [";".join(x) for x in flags]

    return df

def _cluster_description1(st, mz, intens):
    st_min, st_max = (np.min(st), np.max(st))
    mz_min, mz_max = (np.min(mz), np.max(mz))
    area = np.sum(intens)

    return {'st_min': st_min, 'st_max': st_max, 'mz_min': mz_min, 'mz_max': mz_max, 'area': area}
    # later on include ... raggedness, area, etc

    # fdata.update({'st': st, 'I_raw': intens, 'mz': mz})
    # idx_maxI = np.argmax(intens)
    # qc['icent'] = idx_maxI
    #
    # mz_maxI = mz[idx_maxI]
    # mz_min = np.min(mz)
    # mz_max = np.max(mz)
    #
    # rt_maxI = st[idx_maxI]
    # rt_min = np.min(st)
    # rt_max = np.max(st)
    #
    # st_span = rt_max - rt_min
    #
    # ppm = (mz_max - mz_min) / mz_maxI * 1e6
    # qc['ppm'] = ppm
    # descr.update(
    #     {'mzMaxI': mz_maxI, 'rtMaxI': rt_maxI, 'st_span': st_span, 'mz_min': mz_min, 'mz_max': mz_max, 'rt_min': rt_min,
    #      'rt_max': rt_max, })
    #
    # if idx_maxI < 2 or idx_maxI > len(idx) - 2:
    #     return _fail('icent: Peak is not centered', qc, quant, descr, fdata)
    #
    #
    # if ppm > qc_par['ppm']:
    #     return _fail('m/z variability of a feature (ppm)', qc, quant, descr, fdata)
    #
    # # gaps or doublets in successive scan ids
    # sid = Xf[IDX_SCAN_NORM, idx]
    # fdata.update({'sid': sid})
    # sid_diff = np.diff(sid)
    # sid_gap = (np.sum(sid_diff[sid_diff > 1])) / (iLen - 1)  # + np.sum(np.abs(sid_diff[(sid_diff < 1)]))
    # qc['sId_gap'] = sid_gap
    #
    # if sid_gap > self.qc_par['sId_gap']:
    #     return _fail('sId_gap: scan ids not consecutive (%)', qc, quant, descr, fdata)
    #
    # # minor smoothing - minimum three data points - and  padding
    # ysm = np.convolve(intens, np.ones(3) / 3, mode='valid')
    # ysm = np.concatenate(([intens[0]], ysm, [intens[-1]]))
    # xsm = st
    #
    # # I min-max scaling and bline correction
    # intens_max = np.max(intens)
    # intens = intens / intens_max
    # ysm_max = np.max(ysm)
    # ysm = ysm / ysm_max
    # # bline correction of smoothed signal
    # # if smoothed signal has less than three data points, then this won't work
    # # minimum last points minu
    # y_bl = ((np.min(ysm[-3:]) - np.min(ysm[0:3])) / (xsm[-1] - xsm[0])) * (xsm - np.min(xsm)) + np.min(ysm[0:3])
    # ybcor = ysm - y_bl
    #
    # bcor_argmax = np.argmax(ybcor)
    # bcor_max = np.max(ybcor * ysm_max)
    #
    # descr.update({'mzMaxI': mz[bcor_argmax], 'rtMaxI': st[bcor_argmax]})
    #
    # fdata.update({'I_sm_bline': ybcor * ysm_max, 'I_smooth': ysm * ysm_max, 'mz': mz})
    # nneg = np.sum(ybcor >= (-0.05)) / len(ysm)
    # qc['non_neg'] = nneg
    # if nneg < self.qc_par['non_neg']:
    #     return _fail('nneg: Neg Intensity values', qc, quant, descr, fdata)
    #
    # # signal / noise
    # scans = np.unique(sid)
    # scan_id = np.concatenate([np.where((self.Xf[4] == x) & (self.Xf[5] == 1))[0] for x in scans])
    # noiI = np.median(self.Xf[IDX_INT, scan_id]) if len(scan_id) > 0 else np.quantile(Xf[IDX_INT], 0.8)
    # sino = bcor_max / noiI
    # qc['sino'] = sino
    # if sino < self.qc_par['sino']:
    #     return _fail('sino: s/n below qc threshold', qc, quant, descr, fdata)
    #
    # # intensity variation level
    # idx_maxI = np.argmax(ybcor)
    # sdxdy = (np.sum(np.diff(ybcor[:(idx_maxI + 1)]) < (-0.01)) + np.sum(np.diff(ybcor[(idx_maxI):]) > 0.01)) / (
    #         len(ybcor) - 1)  # this is zero if not ragged (np.diff flips array)
    # qc['raggedness'] = sdxdy  # this is zero if not ragged
    #
    # if qc['raggedness'] > self.qc_par['raggedness']:
    #     return _fail('raggedness: intensity variation level above qc threshold', qc, quant, descr, fdata)
    #
    # # integrals
    # # a = integrate.trapz(ybcor * bcor_max, x=xsm)
    # a_raw = integrate.trapz(fdata['I_raw'], fdata['st'])
    # a_smbl = integrate.trapz(fdata['I_sm_bline'], fdata['st'])
    # quant.update({'raw': a_raw, 'smbl': a_smbl})
    # # include symmetry and monotonicity
    # pl = find_peaks(fdata['I_sm_bline'], distance=10)
    # plle = len(pl[0])
    #
    # if plle > 0:
    #     pp = peak_prominences(fdata['I_sm_bline'], pl[0])
    #     if len(pp[0]) > 0:
    #         aa = str(np.round(pp[0][0]))
    #     else:
    #         aa = 0
    # else:
    #     # pl = -1
    #     aa = -1
    # descr.update({'npeaks': plle, 'pprom': aa, })
    #
    # od = {'flev': 3, 'qc': qc, 'quant': quant, 'descr': descr, 'fdata': fdata}
    # return od

def score_peak(d):

    score = 1.0
    penalties = {}

    # --- helpers ---
    def clip01(x):
        return max(0.0, min(1.0, x))

    # --- 1. m/z stability (strong penalty) ---
    ppm = d.get("ppm_wad", d.get("ppm", np.nan))
    if not np.isnan(ppm):
        p = clip01((ppm - 5) / 25)  # ~0 below 5 ppm, bad above 30
        penalties["ppm"] = p * 0.4
        score -= penalties["ppm"]

    # --- 2. scan continuity ---
    sid_gap = d.get("sId_gap", 0)
    p = clip01((sid_gap - 0.05) / 0.3)
    penalties["sid_gap"] = p * 0.3
    score -= penalties["sid_gap"]

    # --- 3. edge behavior (very important) ---
    edge = d.get("edge_max", np.nan)
    if not np.isnan(edge):
        p = clip01((edge - 0.1) / 0.5)
        penalties["edge"] = p * 0.5
        score -= penalties["edge"]

    # --- 4. raggedness ---
    rag = d.get("raggedness", np.nan)
    if not np.isnan(rag):
        p = clip01((rag - 0.1) / 0.4)
        penalties["ragged"] = p * 0.4
        score -= penalties["ragged"]

    # --- 5. multi-peak structure ---
    npeaks = d.get("n_major_peaks", 1)
    valley = d.get("min_valley_ratio", 1)

    if npeaks >= 2:
        if valley < 0.5:
            penalties["split"] = 0.5
        elif valley < 0.8:
            penalties["shoulder"] = 0.25
        else:
            penalties["multi"] = 0.15
        score -= sum(penalties[k] for k in penalties if k in ["split","shoulder","multi"])

    # --- 6. too few points ---
    n = d.get("n", 0)
    if n < 5:
        penalties["n"] = 0.4
    elif n < 8:
        penalties["n"] = 0.15
    else:
        penalties["n"] = 0.0
    score -= penalties["n"]

    # --- 7. apex position (soft) ---
    apex_idx = d.get("apex_idx", None)
    if apex_idx is not None and n > 0:
        left = apex_idx
        right = n - apex_idx - 1
        if min(left, right) < 2:
            penalties["apex_edge"] = 0.2
            score -= penalties["apex_edge"]

    score = clip01(score)
    return score, penalties


def _cluster_summary(clusters, Xf, qc_par, exhaustive=False):

    cl_desc = {'pass': {}, 'fail': {}} # pass

    map_key = {True: 'pass', False: 'fail'}

    # cl_qcf = {} # fail
    for k, cl_idx in clusters.items():

        st = Xf[IDX_ST, cl_idx]
        intens = Xf[IDX_INT, cl_idx]
        mz = Xf[IDX_MZ, cl_idx]
        # sid = Xf[IDX_SCAN_NORM, cl_idx]

        out = _cluster_checkpoint1_pass_fail(cl_idx=cl_idx, Xf=Xf, qc_par=qc_par)

        if exhaustive:
            desc = _cluster_description(st, mz, intens)
            aps = out[1] | desc
        else:
            if out[0]:
                desc = _cluster_description(st, mz, intens)
                # label, flags = classify_one(desc)
                score, penalties = score_peak(desc)
                aps = out[1] | desc | {'qc_score': score, 'penalties': penalties}
            else:
                aps = out[1]

        cl_desc[map_key[out[0]]].update({k: aps})
    return cl_desc

def _per_cluster_summary(st, intens, mz, sid, qc_par):

    passed, chkp = _cluster_checkpoint1_pass_fail1(st=st, intens=intens, mz=mz, sid=sid, qc_par=qc_par)

    if not passed:
        return None

    desc = _cluster_description(st, mz, intens)

    score, penalties = score_peak(desc)

    return chkp | desc | {'qc_score': score, 'penalties': penalties}