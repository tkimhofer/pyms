import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from msmate.core.types import DBSCANParams
from msmate.isotopes.grouping import IsotopeFinder
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import functools
import sys
from tqdm import tqdm
import multiprocessing as mp

mp.set_start_method("fork", force=True)

### SCORING FUNCTIONS FOR EACH PARAMETER RUN

def weighted_score(df):
    # high feature sores - best results?
    # not really, prevent less ideal shapes of low resolved features pushing down score
    # therefore: feature height-weighted qc_score!
    w = np.log1p(df['height']).clip(lower=0)
    if w.sum() == 0:
        return 0.
    return np.average(df['qc_score'], weights=w)

# def valid_feature_fraction(df, qualiy_thres=0.3, noise_level=1000):
#     # counting the number of valid features (higher nb of high quality features)
#     n_valid = len(np.where((df['qc_score'] > qualiy_thres) & (df['height'] > noise_level))[0])
#     return n_valid / df.shape[0]

def plausible_isotope_patterns(df, noise_level=1000):
    iso = df["iso_label"]

    m_peaks = df[iso.isna() | iso.eq("M+0")].copy()

    n_carbons = m_peaks["apex_mz"] / 13.0
    prob_m1 = n_carbons * 0.011

    expected_m1 = (m_peaks["height"] * prob_m1) > (noise_level * 3)
    expected_m2 = (m_peaks["height"] * (prob_m1 ** 2 / 2)) > (noise_level * 3)

    total_expected = expected_m1.sum() + expected_m2.sum()

    total_actual = df["iso_label"].fillna("").str.contains(r"M\+[1-9]").sum()

    if total_expected == 0:
        return 1.0

    return min(total_actual / total_expected, 1.0)



def fragmentation_penalty(df, min_points=5):
    # no peak splitting
    if df.empty:
        return 0.0

    small = (df["n"] < min_points).mean()
    border = df.get("apex_at_edge", False)
    if not isinstance(border, bool):
        border = border.mean()

    penalty = 0.5 * small + 0.5 * border
    return 1.0 - min(penalty, 1.0)

def noise_penalty(df, noise_level=1000):
    if df.empty:
        return 0.0

    weak = (df["height"] < noise_level).mean()
    bad = (df["qc_score"] < 0.2).mean()

    penalty = 0.6 * weak + 0.4 * bad
    return 1.0 - min(penalty, 1.0)

def mz_precision_score(df, ppm_good=5, ppm_bad=20):
    if df.empty:
        return 0.0

    ppm = df["ppm"].clip(lower=0)

    score = 1 - ((ppm - ppm_good) / (ppm_bad - ppm_good)).clip(0, 1)
    w = np.log1p(df["height"].clip(lower=0))

    if w.sum() == 0:
        return score.mean()

    return np.average(score, weights=w)


# CALC SINGLE SCORE FOR EACH RUN FROM SCORING FUNCTIONS
def score_run_df(df, dbs_pars, target_valid, noise_level):
    weighted_qc = weighted_score(df)

    feature_mask = (
            (df["qc_score"] > 0.3) &
            (df["height"] > noise_level)
    )

    valid_fraction = feature_mask.mean()
    n_valid = feature_mask.sum()

    if target_valid is None:
        coverage = 1.0
    else:
        coverage = min(n_valid / target_valid, 1.0)

    isotope = plausible_isotope_patterns(df, noise_level=noise_level)
    frag = fragmentation_penalty(df)
    noise = noise_penalty(df)
    mz_prec = mz_precision_score(df)

    run_score = (
        # geometric mean as multiplicative
            weighted_qc ** 0.2 *
            valid_fraction ** 0.15 *
            coverage ** 0.15 *
            isotope ** 0.30 *
            noise ** 0.10 *
            mz_prec ** 0.05 *
            frag ** 0.05
    )

    return {
        'n_valid': n_valid,
        'pars': dbs_pars.__dict__,
        'run_score': run_score,
        'weighted_qc': weighted_qc,
        'valid_Fraction': valid_fraction,
        'coverage': coverage,
        'isotope': isotope,
        'noise': noise,
        'mz_precision': mz_prec
    }


# FEATURE DETECTION FOR SINGLE PARAMETER SET
def run_single_experiment(p_tuple, exp, swin, target_valid=None, noise_level=1000):
    index, p = p_tuple
    dbs_pars = DBSCANParams(**p)

    # Logic from your loop
    cluster_description = exp.get_features(scan_window=swin, dbs_pars=dbs_pars)

    passed = cluster_description["pass"]
    n = len(passed)

    if n == 0:
        return index, {
            'n_valid': 0,
            'pars': dbs_pars.__dict__,
            'run_score': 0.,
            'weighted_qc': 0.,
            'valid_Fraction': 0.,
            'coverage': 0.,
            'isotope': 0.,
            'noise': 0.,
            'mz_precision': 0.

        }, None

    isot = IsotopeFinder(passed)
    df = isot.find_patterns()
    df["run_id"] = index

    score_dict = score_run_df(df, dbs_pars, target_valid, noise_level)

    return index, score_dict, df


# DEFINE PARAMETER SEARCH GRID AND RUN FEATURE DETECTION FOR EACH PARAMETER SET
def score_runs(exp, swin):

    pars = dict(
        eps=[1.1],
        min_samples=[3,5],
        st_sig=[1],
        mz_sig=np.geomspace(3, 100, 10),
        # st_sig=[0.5, 1, 2, 3],

        q_noise=[0.8, 0.85, 0.90, 0.95, 0.98, 0.99],
    )

    keys = pars.keys()
    grid = [dict(zip(keys, v)) for v in product(*pars.values())]
    # len(grid)

    # run_single_experiment(p_tuple=(1, grid[0]), exp=exp, swin=swin)

    # Use partial to 'freeze' the scan_window argument
    worker = functools.partial(run_single_experiment, swin=swin, exp=exp)

    res = {}
    feats = []
    # enumerate(grid) provides the (index, params) tuples
    with ProcessPoolExecutor(max_workers=7) as executor:
        # map returns results in the order of the grid
        futures = {executor.submit(worker, item): item[0] for item in enumerate(grid)}

        for fut in tqdm(as_completed(futures),
                        total=len(grid),
                        desc="Tuning DBSCAN",
                        file=sys.stdout,
                        leave=True):
            index, data, feat_df = fut.result()
            res[index] = data
            feats.append(feat_df)

    #### check best values, extend if necessary:
    runs = pd.DataFrame(res).T.sort_values('run_score', ascending=False)

    best_mz_sig = runs.iloc[0].pars['mz_sig']

    # if best mz_sig is at upper boundary, expand upward
    if best_mz_sig == pars['mz_sig'][-1]:
        mz_grid = np.geomspace(best_mz_sig, best_mz_sig * 5, 8)
        print('mz_sig value higher than grid values')

    # if best mz_sig is at lower boundary, expand downward
    if best_mz_sig == pars['mz_sig'][0]:
        mz_grid = np.geomspace(best_mz_sig / 5, best_mz_sig, 8)
        print('mz_sig value lower than grid values')

    # TODO: re-run with extended par set

    features = pd.concat(feats).sort_values(['apex_st', 'apex_mz', 'run_id'])
    cord = ['run_id', 'apex_st', 'apex_mz', 'qc_score',]
    cord1 = cord + [x for x in features.columns if x not in cord]
    features = features.loc[:,cord1]

    return runs, features


# COMBINE RUN INFORMATION FOR EACH PARAMETER SET: CALCULATE FEATURE CONFIDENCE VALUE
# def score_stability(features, runs, st_scale=2.0, ppm_scale=5.0,):
#
#     '''
#         group and score signals with about `st_scale` (eg 5 ppm) mz and `ppm_scale` (eg 2 seconds) scan time variation
#     '''
#
#     features = features.copy()
#     runs = runs.copy()
#
#     runs["run_score"] = (
#         pd.to_numeric(runs["run_score"], errors="coerce")
#         .fillna(0.0)
#         .clip(lower=0)
#     )
#     total_score = runs["run_score"].sum()
#
#     if features.empty or total_score == 0:
#         return pd.DataFrame(), features
#
#     X = np.column_stack([
#         features["apex_st"].to_numpy() / st_scale,
#         np.log(features["apex_mz"].to_numpy()) * 1e6 / ppm_scale
#     ])
#
#     labels = DBSCAN(
#         eps=1.0,
#         min_samples=1,
#         metric="euclidean",
#         algorithm="kd_tree",
#         n_jobs=1
#     ).fit_predict(X)
#
#     features["consensus_id"] = labels
#     features["rid_fid"] = (
#             features["run_id"].astype(str) + "_" + features.index.astype(str)
#     )
#
#     out = []
#
#     for cid, sub in features.groupby("consensus_id"):
#         run_ids = sub["run_id"].unique()
#         detected_run_score = runs.loc[run_ids, "run_score"].sum()
#
#         stability = detected_run_score / total_score
#
#         # collapse duplicate detections per run
#         per_run_quality = []
#         per_run_weights = []
#
#         for rid, s in sub.groupby("run_id"):
#             w_feat = np.log1p(s["height"].clip(lower=0))
#
#             if w_feat.sum() == 0:
#                 q = s["qc_score"].mean()
#             else:
#                 q = np.average(s["qc_score"], weights=w_feat)
#
#             per_run_quality.append(q)
#             per_run_weights.append(runs.loc[rid, "run_score"])
#
#         intrinsic_quality = np.average(
#             per_run_quality,
#             weights=np.asarray(per_run_weights) + 1e-12
#         )
#
#         # consistency: real features should have stable mz/rt across runs
#         rt_sd = sub["apex_st"].std(ddof=0)
#         mz_ppm_sd = (
#                 np.std(np.log(sub["apex_mz"].to_numpy()), ddof=0) * 1e6
#         )
#
#         rt_consistency = np.exp(-rt_sd / st_scale)
#         mz_consistency = np.exp(-mz_ppm_sd / ppm_scale)
#         consistency = np.sqrt(rt_consistency * mz_consistency)
#
#         confidence = (
#                 stability ** 0.45 *
#                 intrinsic_quality ** 0.40 *
#                 consistency ** 0.15
#         )
#
#         out.append({
#             "consensus_id": cid,
#             "confidence": confidence,
#             "stability": stability,
#             "intrinsic_quality": intrinsic_quality,
#             "consistency": consistency,
#             "n_detections": len(sub),
#             "n_runs": len(run_ids),
#             "mz_center": np.average(
#                 sub["apex_mz"],
#                 weights=np.log1p(sub["height"].clip(lower=0))
#             ),
#             "rt_center": np.average(
#                 sub["apex_st"],
#                 weights=np.log1p(sub["height"].clip(lower=0))
#             ),
#             "mz_min": sub["apex_mz"].min(),
#             "mz_max": sub["apex_mz"].max(),
#             "rt_min": sub["apex_st"].min(),
#             "rt_max": sub["apex_st"].max(),
#             "rt_sd": rt_sd,
#             "mz_ppm_sd": mz_ppm_sd,
#             "members": sub["rid_fid"].tolist()
#         })
#
#     consensus = pd.DataFrame(out).sort_values(
#         "confidence", ascending=False
#     )
#
#     return consensus, features

def score_stability_fast(features, runs, st_scale=2.0, ppm_scale=5.0):
    features = features.copy()
    runs = runs.copy()

    runs["run_score"] = (
        pd.to_numeric(runs["run_score"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0)
    )

    total_score = runs["run_score"].sum()

    if features.empty or total_score == 0:
        return pd.DataFrame(), features

    X = np.column_stack([
        features["apex_st"].to_numpy() / st_scale,
        np.log(features["apex_mz"].to_numpy()) * 1e6 / ppm_scale
    ])

    features["consensus_id"] = DBSCAN(
        eps=1.0,
        min_samples=1,
        metric="euclidean",
        algorithm="kd_tree",
        n_jobs=1
    ).fit_predict(X)

    features["w_height"] = np.log1p(features["height"].clip(lower=0))
    features["run_score"] = features["run_id"].map(runs["run_score"])
    features["mz_log"] = np.log(features["apex_mz"])
    features["qc_x_w"] = features["qc_score"] * features["w_height"] # heigh-weighted qc score of a feature

    # per consensus + run: collapse fragmented detections from same run
    gr = features.groupby(["consensus_id", "run_id"], sort=False)

    per_run = gr.agg(
        qc_x_w_sum=("qc_x_w", "sum"),
        w_sum=("w_height", "sum"),
        run_score=("run_score", "first"),
    ).reset_index()

    per_run["q_run"] = np.where(
        per_run["w_sum"] > 0,
        per_run["qc_x_w_sum"] / per_run["w_sum"], # averaged heigh-weighted qc score of a feature
        np.nan
    )

    per_run["q_x_runscore"] = per_run["q_run"] * per_run["run_score"] # feature quality times x run score

    # consensus-level stability and intrinsic quality
    c1 = per_run.groupby("consensus_id", sort=False).agg(
        detected_run_score=("run_score", "sum"),
        q_x_runscore=("q_x_runscore", "sum"),  # sum of run-score x feature quality across all runs for single consensus feature
        n_runs=("run_id", "nunique"),
    )

    c1["stability"] = c1["detected_run_score"] / total_score # fraction of score across all runs where feature was detected
    c1["intrinsic_quality"] = c1["q_x_runscore"] / ( # feature quality averaged over all runs - not run specific
        c1["detected_run_score"] + 1e-12
    )


    ### feature descriptors, selecting where qc_score is max
    feat1 = features.reset_index(drop=True).copy()

    # number of detections per consensus feature
    n_det = (
        feat1
        .groupby("consensus_id", sort=False)
        .size()
        .rename("n_detections")
    )

    # row index of best feature per consensus_id
    best_idx = (
        feat1
        .groupby("consensus_id", sort=False)["qc_score"]
        .idxmax()
    )

    # full feature row where qc_score is max
    c2 = (
        feat1
        .loc[best_idx]
        .set_index("consensus_id")
    )

    # add n_detections
    c2 = c2.join(n_det)


    #
    # feat1 = features.copy()
    # feat1.reset_index(drop=True, inplace=True)
    #
    # test = feat1.loc[
    #     feat1.groupby('consensus_id')['qc_score'].idxmax()
    # ]
    #
    # # feature-level consensus summaries
    # c2 = features.groupby("consensus_id", sort=False).agg(
    #     n_detections=("consensus_id", "size"),
    #     mz_min=("apex_mz", "min"),
    #     mz_max=("apex_mz", "max"),
    #     rt_min=("apex_st", "min"),
    #     rt_max=("apex_st", "max"),
    #     rt_sd=("apex_st", lambda x: x.std(ddof=0)),
    #     mz_log_sd=("mz_log", lambda x: x.std(ddof=0)),
    #     # w_sum=("w_height", "sum"),
    #     # mz_weighted_sum=("apex_mz", lambda x: np.nan),  # placeholder
    # )

    # weighted centers separately, faster/cleaner
    # features["mz_x_w"] = features["apex_mz"] * features["w_height"]
    # features["rt_x_w"] = features["apex_st"] * features["w_height"]

    # centers = features.groupby("consensus_id", sort=False).agg(
    #     mz_x_w=("mz_x_w", "sum"),
    #     rt_x_w=("rt_x_w", "sum"),
    #     w_sum=("w_height", "sum"),
    # )
    #
    # c2["mz_center"] = centers["mz_x_w"] / centers["w_sum"]
    # c2["rt_center"] = centers["rt_x_w"] / centers["w_sum"]
    # c2["mz_ppm_sd"] = c2["mz_log_sd"] * 1e6

    consensus = c1.join(c2)

    # rt_consistency = np.exp(-consensus["rt_sd"] / st_scale)
    # mz_consistency = np.exp(-consensus["mz_ppm_sd"] / ppm_scale)
    #
    # consensus["consistency"] = np.sqrt(rt_consistency * mz_consistency)

    consensus["confidence"] = (
        consensus["stability"] ** 0.5 *
        consensus["intrinsic_quality"] ** 0.50
        # consensus["consistency"] ** 0.15
    )

    consensus = (
        consensus
        .reset_index()
        .sort_values("confidence", ascending=False)
    )

    front_cols = [
        "consensus_id",
        "confidence",
        "stability",
        "intrinsic_quality",
        # "consistency",
        "n_runs",
        "n_detections",
        "apex_mz",
        "apex_st",
        # "mz_ppm_sd",
        # "rt_sd",
    ]

    other_cols = [c for c in consensus.columns if c not in front_cols]

    consensus = consensus.loc[:, front_cols + other_cols]

    return consensus, features


