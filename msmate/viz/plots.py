from typeguard import typechecked
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from itertools import compress
from typing import Union, Optional

from msmate.core.types import  ScanWindow, IDX_MZ, IDX_INT, IDX_ST, IDX_SCAN_ORI, IDX_SCAN_NORM, IDX_CLUSTER
from msmate.processing.chromatograms import _xic
from msmate.utils.filters import _window_mz_rt


def _tick_conv(X):
    """Conversion time dimension from seconds to minutes."""
    V = X / 60
    return ["%.2f" % z for z in V]

@typechecked
class PlottingAccessor:
    """ Base class defining visualisation modalities for MS data.

    Note that this class is designed for use with `MsExp`.
    """

    def __init__(self, experiment):
        self.exp = experiment

    def chromatogram(
            self,
            tmin: Optional[float] = None,
            tmax: Optional[float] = None,
            ctype=None,
            xic_mz=None,
            xic_ppm: float = 10,
    ):
        """Plot TIC, BPC and/or XIC chromatograms."""

        df_l1 = self.exp.dfd[self.exp.ms0string]

        if df_l1.shape[0] == 0:
            raise ValueError("MS level 1 dataframe does not exist or is empty.")

        if ctype is None:
            ctype = ["tic", "bpc"]
        elif isinstance(ctype, str):
            ctype = [ctype]

        ctype = [x.lower() for x in ctype]

        valid = {"tic", "bpc", "xic"}
        unknown = set(ctype) - valid
        if unknown:
            raise ValueError(f"Unknown ctype(s): {unknown}. Use 'tic', 'bpc', or 'xic'.")

        xic = "xic" in ctype
        tic = "tic" in ctype
        bpc = "bpc" in ctype

        if xic and xic_mz is None:
            raise ValueError("xic_mz must be provided when ctype includes 'xic'.")

        if tmin is None:
            tmin = df_l1["Rt"].min()
        if tmax is None:
            tmax = df_l1["Rt"].max()

        df_plot = df_l1.loc[
            (df_l1["Rt"] >= tmin) & (df_l1["Rt"] <= tmax)
            ]

        fig, ax1 = plt.subplots(1, 1)

        if xic:
            xx, xy = _xic(
                self.exp,
                xic_mz,
                xic_ppm,
                rt_min=tmin,
                rt_max=tmax,
            )
            ax1.plot(
                xx,
                xy.astype(float),
                label=f"XIC {xic_mz} m/z ± {xic_ppm} ppm",
            )

        if tic:
            ax1.plot(
                df_plot["Rt"],
                df_plot["tic"].astype(float),
                label="TIC",
            )

        if bpc:
            ax1.plot(
                df_plot["Rt"],
                df_plot["bpi"].astype(float),
                label="BPC",
            )

        ax1.text(
            1.04,
            0,
            self.exp.fname,
            rotation=90,
            fontsize=6,
            transform=ax1.transAxes,
        )

        ax1.set_xlabel(r"Scantime (s)")
        ax1.set_ylabel(r"Count")

        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())

        tick_loc = np.arange(
            np.floor(tmin / 60),
            np.ceil(tmax / 60) + 1,
            2,
            dtype=float,
        ) * 60

        ax2.set_xticks(tick_loc)
        ax2.set_xticklabels(_tick_conv(tick_loc))
        ax2.set_xlabel(r"Scantime (min)")

        ax1.legend(loc="best")

        fig.tight_layout()

        return fig, ax1, ax2


    def _noiseT(self, p, X=None, local=True):
        """Calculation of a noise intensity threshold using all data points or windowed data points."""
        if local:
            # self.noise_thres = np.quantile(X, q=p)
            idxN = X[IDX_INT] > np.quantile(X, q=p)
            idc_above = np.where(idxN)[0]
            idc_below = np.where(~idxN)[0]
            return (idc_below, idc_above)
        else:
            self.noise_thres = np.quantile(self.xrawd[self.ms0string], q=p)
        return self.noise_thre

        # idxN = X[IDX_INT] > self.noise_thres
        # idc_above = np.where(idxN)[0]
        # idc_below = np.where(~idxN)[0]
        # return (idc_below, idc_above)

    def st_mz_scatter(
        self,
        q_noise: float,
        selection: Union[ScanWindow, None],
        qcm_local: bool = True,
    ):
        if selection is None:
            Xsub = self.exp.xrawd[self.exp.ms0string]
        else:
            Xsub = _window_mz_rt(
                self.exp.xrawd[self.exp.ms0string],
                selection=selection,
            )

        idc_below, idc_above = self._noiseT(
            p=q_noise,
            X=Xsub,
            local=qcm_local,
        )

        fig, ax = plt.subplots()

        ax.text(
            1.25, 0,
            str(self.exp.dpath),
            rotation=90,
            fontsize=6,
            transform=ax.transAxes,
        )

        if len(idc_below) > 0:
            ax.scatter(
                Xsub[IDX_ST, idc_below],
                Xsub[IDX_MZ, idc_below],
                s=0.1,
                c="gray",
                alpha=0.5,
            )

        if len(idc_above) > 0:
            vals = Xsub[IDX_INT, idc_above]
            pos = vals > 0

            if np.any(pos):
                im = ax.scatter(
                    Xsub[IDX_ST, idc_above][pos],
                    Xsub[IDX_MZ, idc_above][pos],
                    c=vals[pos],
                    s=5,
                    cmap=plt.colormaps["viridis"],
                    norm=LogNorm(),
                )
                fig.colorbar(im, ax=ax, label="Intensity")

        ax.set_xlabel("Scan time [sec]")
        ax.set_ylabel("m/z")
        ax.yaxis.offsetText.set_visible(False)

        tmax = np.max(Xsub[IDX_ST])
        tmin = np.min(Xsub[IDX_ST])

        tick_loc = np.linspace(tmin, tmax, 5)

        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(tick_loc)
        ax2.set_xticklabels(_tick_conv(tick_loc))
        ax2.set_xlabel("[min]")

        fig.tight_layout()

        return fig, (ax, ax2)

    @staticmethod
    def seconds_to_minutes_labels(X):
        """Conversion time dimension from seconds to minutes."""
        V = X / 60
        return ["%.2f" % z for z in V]

    def feature(
            self,
            feature_id,
            mz_pad_ppm: float = 100,
            st_pad_sec: float = 10,
            show_isotopes: bool = False,
            show_boundaries: bool = True,
    ):
        """
        Visualise LC-MS feature with:
            - local RT × m/z cloud
            - chromatographic trace
            - feature highlighting
            - apex / boundaries
            - isotope guides (optional)

        Assumes:
            self.exp.Xff shape = (variables, points)
        """

        #3.0, 5.0, 8.0, 12.0, 22.0, 23.0, 29.0, 34.0, 36.0, 37.0, 38.0,
        # feature_id = 8.

        cl_props = self.exp.cluster_description['pass'][feature_id]

        mz_dist = (cl_props['mz_wmean'] / 1e6) * mz_pad_ppm
        mz_range = (
            cl_props['mz_min'] - mz_dist,
            cl_props['mz_max'] + mz_dist
        )

        st_range = (
            cl_props['st_min'] - st_pad_sec,
            cl_props['st_max'] + st_pad_sec
        )

        mz = self.exp.Xff[IDX_MZ]
        st = self.exp.Xff[IDX_ST]
        inten = self.exp.Xff[IDX_INT]
        cl = self.exp.Xff[IDX_CLUSTER]

        mask = (
                (mz >= mz_range[0]) &
                (mz <= mz_range[1]) &
                (st >= st_range[0]) &
                (st <= st_range[1])
        )

        mask_f = mask & (cl == feature_id)


        fig = plt.figure(figsize=(12, 8))

        gs = GridSpec(
            2,
            3,
            width_ratios=[4, 0.12, 1.2],
            height_ratios=[3, 1.6],
        )

        ax_map = fig.add_subplot(gs[0, 0])
        ax_trace = fig.add_subplot(gs[1, 0], sharex=ax_map)
        cax = fig.add_subplot(gs[0, 1])
        ax_info = fig.add_subplot(gs[:, 2])

        # gs = GridSpec(
        #     2,
        #     2,
        #     width_ratios=[4, 1.2],
        #     height_ratios=[3, 1.6],
        #     hspace=0.08,
        #     wspace=0.12
        # )
        #
        # ax_map = fig.add_subplot(gs[0, 0])
        # ax_trace = fig.add_subplot(gs[1, 0], sharex=ax_map)
        # ax_info = fig.add_subplot(gs[:, 1])

        sc = ax_map.scatter(
            st[mask],
            mz[mask],
            c=inten[mask],
            s=8,
            alpha=0.25,
            cmap='viridis',
            norm=LogNorm(
                vmin=max(np.quantile(inten[mask], 0.05), 1),
                vmax=np.quantile(inten[mask], 0.995)
            ),
            rasterized=True
        )

        # highlighted feature
        ax_map.scatter(
            st[mask_f],
            mz[mask_f],
            c=inten[mask_f],
            s=24,
            edgecolor='black',
            linewidth=0.4,
            cmap='viridis',
            norm=LogNorm(
                vmin=max(np.quantile(inten[mask_f], 0.05), 1),
                vmax=np.max(inten[mask_f])
            ),
            zorder=10
        )


        if show_boundaries:
            rect = Rectangle(
                (cl_props['st_min'], cl_props['mz_min']),
                cl_props['st_max'] - cl_props['st_min'],
                cl_props['mz_max'] - cl_props['mz_min'],
                fill=False,
                lw=2,
                linestyle='--'
            )

            ax_map.add_patch(rect)

        #
        # ax_map.scatter(
        #     cl_props['apex_st'],
        #     cl_props['apex_mz'],
        #     s=120,
        #     marker='x',
        #     linewidth=2.5,
        #     zorder=20
        # )


        if show_isotopes:

            iso_shift = 1.003355

            for i in range(1, 4):
                ax_map.axhline(
                    cl_props['mz_wmean'] + (iso_shift * i),
                    linestyle=':',
                    alpha=0.4
                )

        st_f = st[mask_f]
        int_f = inten[mask_f]

        order = np.argsort(st_f)

        ax_trace.plot(
            st_f[order],
            int_f[order],
            lw=2
        )

        ax_trace.scatter(
            st_f,
            int_f,
            s=20,
            zorder=5
        )

        # apex line
        ax_trace.axvline(
            cl_props['apex_st'],
            linestyle='--',
            alpha=0.6
        )

        if show_boundaries:
            ax_trace.axvspan(
                cl_props['st_min'],
                cl_props['st_max'],
                alpha=0.12
            )

        ax_map.set_ylabel("m/z")
        ax_trace.set_ylabel("Intensity")
        ax_trace.set_xlabel("Retention time (sec)")

        ax_map.set_title(
            f"Feature {feature_id}",
            loc='left',
            fontsize=14,
            fontweight='bold'
        )

        ax_info.axis('off')

        qc_txt = (
            f"Feature ID\n"
            f"{feature_id}\n\n"
            f"QC Score\n"
            f"{cl_props['qc_score']:.2f}\n\n\n"
            
            f"n Points\n"
            f"{cl_props.get('n', 'NA')}\n\n"
            f"m/z\n"
            f"{cl_props['mz_wmean']:.4f} ({((cl_props['mz_max'] - cl_props['mz_min']) / cl_props['mz_wmean']) * 1e6:.2f} ppm)\n\n"
            f"ST apex (span)\n"
            f"{cl_props['apex_st']:.2f} ({cl_props['st_span']:.2f}) sec\n\n"
            f"Sum Counts\n"
            f"{cl_props['sum_intensity']:.2e}\n\n\n"
            
            f"n major Peaks\n"
            f"{cl_props['n_major_peaks']:.0f}\n\n"
            f"Raggedness\n"
            f"{cl_props['raggedness']:.2f}\n\n"
            f"Gaps in Scan-Id Sequence\n"
            f"{cl_props['sId_gap']:.2f}\n\n"
            f"Edge penalties\n"
            f"left: {cl_props['edge_left']:.2f}, right:  {cl_props['edge_right']:.2f}\n\n"
        )

        ax_info.text(
            0.08,
            0.92,
            qc_txt,
            va='top',
            # family='monospace',
            fontsize=10
        )

        # cbar = fig.colorbar(
        #     sc,
        #     ax=ax_map,
        #     fraction=0.025,
        #     pad=0.02
        # )
        #
        # cbar.set_label("Intensity")
        fig.colorbar(sc, cax=cax)

        ax_map.spines[['top', 'right']].set_visible(False)
        ax_trace.spines[['top', 'right']].set_visible(False)

        plt.setp(ax_map.get_xticklabels(), visible=False)

        fig.tight_layout()

        return fig

    def feature1(self, feature_id, mz_pad_ppm:float=100, st_pad_sec:float=10, show_isotopes:bool=True, show_boundaries:bool=True):

        # feature_id = list(self.exp.cluster_description['pass'].keys())[1]
        cl_props = self.exp.cluster_description['pass'][feature_id]

        mz_dist = (cl_props['mz_wmean']/1e6) * mz_pad_ppm
        mz_range = (cl_props['mz_min'] - mz_dist, cl_props['mz_max'] + mz_dist)

        st_range = (cl_props['st_min'] - st_pad_sec, (cl_props['st_max'] + st_pad_sec))

        mz = self.exp.Xff[IDX_MZ]
        st = self.exp.Xff[IDX_ST]
        ct = self.exp.Xff[IDX_INT]
        cl = self.exp.Xff[IDX_CLUSTER]
        mask = (mz >= mz_range[0]) & (mz <= mz_range[1]) & (st >= st_range[0]) & (st <= st_range[1])

        plt.scatter(x=st[mask], y=mz[mask], c=ct[mask])

        mask_f = mask | (cl==feature_id)

        plt.plot(st[mask_f], ct[mask_f])
        plt.scatter(x=st[mask_f], y=ct[mask_f])

    def consensus_feature(
            self,
            consensus_id,
            consensus,
            features,
            mz_pad_ppm: float = 100,
            st_pad_sec: float = 10,
            show_boundaries: bool = True,
            show_members: bool = True,
    ):
        """
        Visualise consensus feature from score_stability_fast().

        Parameters
        ----------
        consensus_id:
            ID from consensus["consensus_id"].
        consensus:
            Consensus table returned by score_stability_fast().
        features:
            Feature table returned by score_stability_fast().
        """

        con = consensus.loc[consensus["consensus_id"] == consensus_id]

        if con.empty:
            raise ValueError(f"consensus_id {consensus_id} not found.")

        con = con.iloc[0]

        members = features.loc[
            features["consensus_id"] == consensus_id
            ].copy()

        if members.empty:
            raise ValueError(f"No member features for consensus_id {consensus_id}.")

        mz_center = con["apex_mz"]
        rt_center = con["apex_st"]

        mz_dist = (mz_center / 1e6) * mz_pad_ppm

        mz_range = (
            mz_center - mz_dist,
            mz_center + mz_dist,
        )

        st_range = (
            con["st_min"] - st_pad_sec,
            con["st_max"] + st_pad_sec,
        )

        # raw MS points, not one DBSCAN-labelled result
        X = self.exp.xrawd[self.exp.ms0string]

        mz = X[IDX_MZ]
        st = X[IDX_ST]
        inten = X[IDX_INT]

        mask = (
                (mz >= mz_range[0]) &
                (mz <= mz_range[1]) &
                (st >= st_range[0]) &
                (st <= st_range[1])
        )

        if not np.any(mask):
            raise ValueError("No raw points found in consensus feature window.")

        fig = plt.figure(figsize=(12, 8))

        gs = GridSpec(
            2,
            3,
            width_ratios=[4, 0.12, 1.2],
            height_ratios=[3, 1.6],
        )

        ax_map = fig.add_subplot(gs[0, 0])
        ax_map.set_ylim(mz_range)
        ax_map.set_xlim(st_range)
        ax_map.grid(alpha=0.15)

        ax_trace = fig.add_subplot(gs[1, 0], sharex=ax_map)
        ax_trace.grid(alpha=0.15)
        ax_map.set_xlim(st_range)

        cax = fig.add_subplot(gs[0, 1])
        ax_info = fig.add_subplot(gs[:, 2])

        vals = inten[mask]
        vals_pos = vals[vals > 0]

        if vals_pos.size == 0:
            norm = None
        else:
            norm = LogNorm(
                vmin=max(np.quantile(vals_pos, 0.05), 1),
                vmax=np.quantile(vals_pos, 0.995),
            )

        # member detections across parameter runs
        if show_members:
            for _, m in members.iterrows():
                ax_map.add_patch(
                    Rectangle(
                        (m["st_min"], m["mz_min"]),
                        m["st_max"] - m["st_min"],
                        m["mz_max"] - m["mz_min"],
                        fill=False,
                        lw=0.5,
                        alpha=0.1,
                        linestyle="-",
                        zorder=1,
                        color='grey',
                    )
                )

        # this might be too translucent
        sc = ax_map.scatter(
            st[mask],
            mz[mask],
            c=inten[mask],
            s=8,
            alpha=0.65,
            cmap="viridis",
            norm=norm,
            rasterized=True,
            zorder=30,
        )

        mz_min = con["mz_min"]
        mz_max = con["mz_max"]

        st_min = con["st_min"]
        st_max = con["st_max"]

        # mz_min = members["mz_min"].min()
        # mz_max = members["mz_max"].max()

        # st_min = members["st_min"].min()
        # st_max = members["st_max"].max()

        mask_consensus = (
                (mz >= mz_min) &
                (mz <= mz_max) &
                (st >= st_min) &
                (st <= st_max)
        )


        ax_map.scatter(
            st[mask_consensus],
            mz[mask_consensus],
            c=inten[mask_consensus],
            s=32,
            # edgecolor="black",
            linewidth=0.3,
            cmap="viridis",
            norm=norm,
            zorder=30,
        )

            # ax_map.scatter(
            #     members["apex_st"],
            #     members["apex_mz"],
            #     s=35,
            #     c=members["qc_score"],
            #     edgecolor="black",
            #     linewidth=0.4,
            #     cmap="viridis",
            #     zorder=20,
            #     label="member detections",
            # )


        if show_boundaries:
            rect = Rectangle(
                (st_min, mz_min),
                st_max - st_min,
                mz_max - mz_min,
                fill=False,
                edgecolor="#8b0000",
                lw=1.8,
                # facecolor="darkgrey",
                # edgecolor="black",  # or "none"
                # lw=0.2,
                alpha=0.9,
                linestyle="-",
                zorder=20,
            )
            ax_map.add_patch(rect)

        # XIC-like trace around consensus m/z
        mz_trace_mask = (
                (mz >= mz_min) &
                (mz <= mz_max) &
                (st >= st_min) &
                (st <= st_max)
        )

        trace_df = (
            pd.DataFrame({
                "rt": st[mz_trace_mask],
                "intensity": inten[mz_trace_mask],
            })
            .groupby("rt", as_index=False)["intensity"]
            .sum()
            .sort_values("rt")
        )

        ax_trace.plot(
            trace_df["rt"],
            trace_df["intensity"],
            lw=2,
        )

        ax_trace.scatter(
            trace_df["rt"],
            trace_df["intensity"],
            s=20,
            zorder=5,
        )

        ax_trace.axvline(
            rt_center,
            linestyle="--",
            alpha=0.6,
        )

        if show_boundaries:
            ax_trace.axvspan(
                con["st_min"],
                con["st_max"],
                color="#b22222",
                alpha=0.1,
            )

        ax_map.set_ylabel("m/z")
        ax_trace.set_ylabel("Intensity")
        ax_trace.set_xlabel("Scan time (sec)")

        ax_map.set_title(
            f"Consensus feature {consensus_id}",
            loc="left",
            fontsize=14,
            fontweight="bold",
        )

        ax_map.legend(loc="best", fontsize=8)

        ax_info.axis("off")

        info = (
            f"Consensus ID\n"
            f"{consensus_id}\n\n"

            f"Feature confidence\n"
            f"{100.*con['confidence']:.0f}\n\n"

            f"Stability across parameter sets\n"
            f"{100.*con['stability']:.0f}\n\n"

            f"Intrinsic quality\n"
            f"{100.*con['intrinsic_quality']:.0f}\n\n"

            f"Isotopologue label\n"
            f"{con['iso_label']}\n\n"

            f"n runs / detections\n"
            f"{int(con['n_runs'])} / {int(con['n_detections'])}\n\n"

            f"m/z center\n"
            f"{con['apex_mz']:.5f}\n\n"

            f"Scantime center\n"
            f"{con['apex_st']:.1f} sec\n\n"
            
            f"Peak area\n"
            f"{con['area']:.3e}\n\n"

            f"Feature QC score\n"
            f"{con['qc_score']:.1f}\n\n"

            # f"ST SD\n"
            # f"{con['st_sd']:.2f} sec\n\n"

            f"m/z range\n"
            f"{con['mz_min']:.5f} - {con['mz_max']:.5f}\n\n"

            f"Scantime range\n"
            f"{con['st_min']:.2f} - {con['st_max']:.2f} sec\n"
        )

        ax_info.text(
            0.08,
            0.92,
            info,
            va="top",
            fontsize=10,
        )

        fig.colorbar(sc, cax=cax, label="Raw intensity")

        ax_map.spines[["top", "right"]].set_visible(False)
        ax_trace.spines[["top", "right"]].set_visible(False)

        for ax in [ax_map, ax_trace]:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        plt.setp(ax_map.get_xticklabels(), visible=False)

        fig.tight_layout()

        return fig


    def features(self, selection: Union[ScanWindow, dict, None] = None, lev: int = 3):
        """Visualise feature (peak picked) data.

        Interactive scantime vs m/z plot with features highlighted, m/z and Intensity plot of selected features

        Args:
            selection: m/z and scantime (in sec) window
            lev: Feature level for marking features (`2` or `3`)

        Ex:
            vizpp(selection={'mz_min': 0, 'mz_max': 1500, 'rt_min': 5, 'rt_max': 60})
        """

        if selection is None:
            selection = self.selection

        # TODO:
        # label feature with id in m/z Int plot
        # two panel plot: bottom m/z over st, top: st vs intentsity upon click on feature
        # basically like vis spectrum but including rectangles and all points but those in fgr feature(s) are s=0.1 and c='grey

        # @log
        def _vis_feature(fdict, id, ax=None, add=False):
            # idx = np.where(self.Xf[:, 6] == fdict['id'])[0]
            if isinstance(ax, type(None)):
                fig = plt.figure()
                ax = fig.add_subplot(111)

            # print(id)
            fwhm = self._fwhmBound(str(id), rtDelta=1)  # (ul1, ll1)
            # print(fwhm)
            ax.vlines(fwhm[0], 0, np.max(fdict['fdata']['I_smooth']))
            ax.vlines(fwhm[1], 0, np.max(fdict['fdata']['I_smooth']))
            iid = fdict['qc']['icent']
            ax.scatter(fdict['fdata']['st'][iid], fdict['fdata']['I_sm_bline'][iid], c='red')

            ax.plot(fdict['fdata']['st'], fdict['fdata']['I_raw'], label='raw', linewidth=0.5, color='black',
                    zorder=0)

            if fdict['flev'] == 3:
                ax.plot(fdict['fdata']['st'], fdict['fdata']['I_smooth'], label='smoothed', linewidth=0.5,
                        color='cyan', zorder=0)
                # ax.plot(fdict['fdata']['st'], fdict['fdata']['I_bl'], label='baseline', linewidth=0.5,
                #         color='black', ls='dashed', zorder=0)
                ax.plot(fdict['fdata']['st'], fdict['fdata']['I_sm_bline'], label='smoothed bl-cor',
                        color='black', zorder=0)
            if not add:
                ax.set_title(id + f',  flev: {fdict["flev"]}')
                ax.legend()
            else:
                old = axs[0].get_title()
                ax.set_title(old + '\n' + id + f',  flev: {fdict["flev"]}')
            ax.set_ylabel("Count")
            ax.set_xlabel("Scan time, s")

            data = np.array([list(fdict['qc'].keys()), np.round(list(fdict['qc'].values()), 2)]).T
            column_labels = ["descr", "value"]
            ax.table(cellText=data, colLabels=column_labels, loc='lower right', colWidths=[0.1] * 3)
            return ax

        # which feature - define plot axis boundaries
        Xsub = self._window_mz_rt(self.exp.Xf, selection, allow_none=False, return_idc=False)
        fid_window = np.unique(Xsub[7])
        l2_ffeat = list(compress(self.feat_l2, np.in1d([float(i.split(':')[1]) for i in self.feat_l2], fid_window)))
        l3_ffeat = list(compress(self.feat_l3, np.in1d([float(i.split(':')[1]) for i in self.feat_l3], fid_window)))

        # cm = plt.cm.get_cmap('gist_ncar')
        fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 2]}, figsize=(8, 10),
                                constrained_layout=False)
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        axs[0].set_ylabel("Count", fontweight="bold")
        axs[0].set_xlabel("Scan time, s", fontweight="bold")

        axs[1].set_facecolor('#EBFFFF')
        axs[1].text(1.03, 0, self.dpath, rotation=90, fontsize=6, transform=axs[1].transAxes)

        # fill axis for raw data results
        idcA = Xsub[2] > self.noise_thres
        idc_above = np.where(idcA)[0]
        idc_below = np.where(~idcA)[0]

        cm = plt.cm.get_cmap('rainbow')
        axs[1].scatter(Xsub[3, idc_below], Xsub[1, idc_below], s=0.1, c='gray', alpha=0.5)

        imax = np.log(Xsub[2, idc_above])
        psize = (imax / np.max(imax)) * 5
        im = axs[1].scatter(Xsub[3, idc_above], Xsub[1, idc_above], c=Xsub[2, idc_above], s=psize, cmap=cm,
                            norm=LogNorm())

        # Xf
        # 0: scanIdOri,
        # 1: mz,
        # 2: Int,
        # 3: st,
        # 4: scanIdNorm,
        # 5: noiseBool,
        # 6: st_adj,
        # 7: clMem

        # axs[1].scatter(Xsub[Xsub[:, 4] == 0, 3], Xsub[Xsub[:, 4] == 0, 1], s=0.1, c='gray', alpha=0.5)
        iq = 0
        ii = ''
        if lev < 3:
            l2l3_feat = l2_ffeat + l3_ffeat
            for i in l2l3_feat:
                # print(i)
                fid = float(i.split(':')[1])
                f1 = self.feat[i]
                if (f1['flev'] == 2):
                    a_col = 'red'
                else:
                    a_col = 'green'

                it = np.max(f1['fdata']['I_raw'])
                if it > iq:
                    iq = it
                    ii = i
                fsize = 6
                mz = Xsub[1, Xsub[7] == fid]
                if len(mz) == len(f1['fdata']['st']):
                    # im = axs[1].scatter(f1['fdata']['st'], mz,
                    #                     c=np.log(f1['fdata']['I_raw']), s=5, cmap=cm)
                    axs[1].add_patch(Rectangle(((f1['descr']['rt_min']), f1['descr']['mz_min']), \
                                               ((f1['descr']['rt_max']) - (f1['descr']['rt_min'])), \
                                               (f1['descr']['mz_max'] - f1['descr']['mz_min']), fc='#C2D0D6', \
                                               linestyle="solid", color='grey', linewidth=2, zorder=0,
                                               in_layout=True,
                                               picker=False))
                    axs[1].annotate(i.split(':')[1], (f1['descr']['rt_max'], f1['descr']['mz_max']),
                                    bbox=dict(facecolor=a_col, alpha=0.3, boxstyle='circle', edgecolor='white',
                                              in_layout=False),
                                    wrap=False, picker=True, fontsize=fsize)
            # _vis_feature(fdict=self.feat[ii], id=ii, ax=axs[0])

        if lev == 3:
            for i in l3_ffeat:
                # print(i)
                fid = float(i.split(':')[1])
                f1 = self.feat[i]
                if f1['quant']['raw'] > iq:
                    iq = f1['quant']['raw']
                    ii = i
                a_col = 'green'
                fsize = 10
                mz = Xsub[1, Xsub[7] == fid]
                if len(mz) == len(f1['fdata']['st']):
                    axs[1].add_patch(Rectangle(((f1['descr']['rt_min']), f1['descr']['mz_min']), \
                                               ((f1['descr']['rt_max']) - (f1['descr']['rt_min'])), \
                                               (f1['descr']['mz_max'] - f1['descr']['mz_min']), fc='#C2D0D6', \
                                               linestyle="solid", color='grey', linewidth=2, zorder=0,
                                               in_layout=True,
                                               picker=False))
                    axs[1].annotate(i.split(':')[1], (f1['descr']['rt_max'], f1['descr']['mz_max']),
                                    bbox=dict(facecolor=a_col, alpha=0.3, boxstyle='circle', edgecolor='white',
                                              in_layout=False),
                                    wrap=False, picker=True, fontsize=fsize)
            _vis_feature(fdict=self.feat[ii], id=ii, ax=axs[0])
            # im = axs[1].scatter(f1['fdata']['st'], mz,
            #                     c=np.log(f1['fdata']['I_raw']), s=5, cmap=cm)

        # cbaxes = inset_axes(axs[1], width="30%", height="5%", loc=3)
        # plt.colorbar(im, cax=cbaxes, ticks=[0., 1], orientation='horizontal')

        def p_text(event):
            # print(event.artist)
            ids = str(event.artist.get_text())
            if event.mouseevent.button is MouseButton.LEFT:
                # print('left click')
                # print('id:' + str(ids))
                axs[0].clear()
                axs[0].set_title('')
                _vis_feature(self.feat['id:' + ids], id='id:' + ids, ax=axs[0], add=False)
                event.canvas.draw()
            if event.mouseevent.button is MouseButton.RIGHT:
                # print('right click')
                # print('id:' + str(ids))
                _vis_feature(self.feat['id:' + ids], id='id:' + ids, ax=axs[0], add=True)
                event.canvas.draw()

        cid1 = fig.canvas.mpl_connect('pick_event', p_text)
        axs[1].set_xlabel("Scan time, s")
        axs[1].yaxis.offsetText.set_visible(False)
        axs[1].yaxis.set_label_text("m/z")

    def features_depr(self, selection: Union[ScanWindow, dict, None] = None, lev: int = 3):
        """Visualise feature (peak picked) data.

        Interactive scantime vs m/z plot with features highlighted, m/z and Intensity plot of selected features

        Args:
            selection: m/z and scantime (in sec) window
            lev: Feature level for marking features (`2` or `3`)

        Ex:
            vizpp(selection={'mz_min': 0, 'mz_max': 1500, 'rt_min': 5, 'rt_max': 60})
        """

        if selection is None:
            selection = self.selection

        # TODO:
        # label feature with id in m/z Int plot
        # two panel plot: bottom m/z over st, top: st vs intentsity upon click on feature
        # basically like vis spectrum but including rectangles and all points but those in fgr feature(s) are s=0.1 and c='grey

        # @log
        def _vis_feature(fdict, id, ax=None, add=False):
            # idx = np.where(self.Xf[:, 6] == fdict['id'])[0]
            if isinstance(ax, type(None)):
                fig = plt.figure()
                ax = fig.add_subplot(111)

            # print(id)
            fwhm = self._fwhmBound(str(id), rtDelta=1) #(ul1, ll1)
            # print(fwhm)
            ax.vlines(fwhm[0], 0, np.max(fdict['fdata']['I_smooth']))
            ax.vlines(fwhm[1], 0, np.max(fdict['fdata']['I_smooth']))
            iid=fdict['qc']['icent']
            ax.scatter(fdict['fdata']['st'][iid], fdict['fdata']['I_sm_bline'][iid], c='red')

            ax.plot(fdict['fdata']['st'], fdict['fdata']['I_raw'], label='raw', linewidth=0.5, color='black',
                    zorder=0)

            if fdict['flev'] == 3:
                ax.plot(fdict['fdata']['st'], fdict['fdata']['I_smooth'], label='smoothed', linewidth=0.5,
                        color='cyan', zorder=0)
                # ax.plot(fdict['fdata']['st'], fdict['fdata']['I_bl'], label='baseline', linewidth=0.5,
                #         color='black', ls='dashed', zorder=0)
                ax.plot(fdict['fdata']['st'], fdict['fdata']['I_sm_bline'], label='smoothed bl-cor',
                        color='black', zorder=0)
            if not add:
                ax.set_title(id + f',  flev: {fdict["flev"]}')
                ax.legend()
            else:
                old = axs[0].get_title()
                ax.set_title(old + '\n' + id + f',  flev: {fdict["flev"]}')
            ax.set_ylabel("Count")
            ax.set_xlabel("Scan time, s")

            data = np.array([list(fdict['qc'].keys()), np.round(list(fdict['qc'].values()), 2)]).T
            column_labels = ["descr", "value"]
            ax.table(cellText=data, colLabels=column_labels, loc='lower right', colWidths=[0.1] * 3)
            return ax

        # which feature - define plot axis boundaries
        Xsub = self._window_mz_rt(self.exp.Xf, selection, allow_none=False, return_idc=False)
        fid_window = np.unique(Xsub[7])
        l2_ffeat = list(compress(self.feat_l2, np.in1d([float(i.split(':')[1]) for i in self.feat_l2], fid_window)))
        l3_ffeat = list(compress(self.feat_l3, np.in1d([float(i.split(':')[1]) for i in self.feat_l3], fid_window)))

        # cm = plt.cm.get_cmap('gist_ncar')
        fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 2]}, figsize=(8, 10),
                                constrained_layout=False)
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        axs[0].set_ylabel("Count", fontweight="bold")
        axs[0].set_xlabel("Scan time, s", fontweight="bold")

        axs[1].set_facecolor('#EBFFFF')
        axs[1].text(1.03, 0, self.dpath, rotation=90, fontsize=6, transform=axs[1].transAxes)

        # fill axis for raw data results
        idcA = Xsub[2] > self.noise_thres
        idc_above = np.where(idcA)[0]
        idc_below = np.where(~idcA)[0]

        cm = plt.cm.get_cmap('rainbow')
        axs[1].scatter(Xsub[3, idc_below], Xsub[1, idc_below], s=0.1, c='gray', alpha=0.5)

        imax = np.log(Xsub[2, idc_above])
        psize = (imax / np.max(imax)) * 5
        im = axs[1].scatter(Xsub[3, idc_above], Xsub[1, idc_above], c=Xsub[2, idc_above], s=psize, cmap=cm,
                            norm=LogNorm())

        # Xf
        # 0: scanIdOri,
        # 1: mz,
        # 2: Int,
        # 3: st,
        # 4: scanIdNorm,
        # 5: noiseBool,
        # 6: st_adj,
        # 7: clMem

        # axs[1].scatter(Xsub[Xsub[:, 4] == 0, 3], Xsub[Xsub[:, 4] == 0, 1], s=0.1, c='gray', alpha=0.5)
        iq = 0
        ii = ''
        if lev < 3:
            l2l3_feat = l2_ffeat + l3_ffeat
            for i in l2l3_feat:
                # print(i)
                fid = float(i.split(':')[1])
                f1 = self.feat[i]
                if (f1['flev'] == 2):
                    a_col = 'red'
                else:
                    a_col = 'green'

                it = np.max(f1['fdata']['I_raw'])
                if it > iq:
                    iq = it
                    ii = i
                fsize = 6
                mz = Xsub[1, Xsub[7] == fid]
                if len(mz) == len(f1['fdata']['st']):
                    # im = axs[1].scatter(f1['fdata']['st'], mz,
                    #                     c=np.log(f1['fdata']['I_raw']), s=5, cmap=cm)
                    axs[1].add_patch(Rectangle(((f1['descr']['rt_min']), f1['descr']['mz_min']), \
                                               ((f1['descr']['rt_max']) - (f1['descr']['rt_min'])), \
                                               (f1['descr']['mz_max'] - f1['descr']['mz_min']), fc='#C2D0D6', \
                                               linestyle="solid", color='grey', linewidth=2, zorder=0,
                                               in_layout=True,
                                               picker=False))
                    axs[1].annotate(i.split(':')[1], (f1['descr']['rt_max'], f1['descr']['mz_max']),
                                    bbox=dict(facecolor=a_col, alpha=0.3, boxstyle='circle', edgecolor='white',
                                              in_layout=False),
                                    wrap=False, picker=True, fontsize=fsize)
            # _vis_feature(fdict=self.feat[ii], id=ii, ax=axs[0])

        if lev == 3:
            for i in l3_ffeat:
                # print(i)
                fid = float(i.split(':')[1])
                f1 = self.feat[i]
                if f1['quant']['raw'] > iq:
                    iq = f1['quant']['raw']
                    ii = i
                a_col = 'green'
                fsize = 10
                mz = Xsub[1, Xsub[7] == fid]
                if len(mz) == len(f1['fdata']['st']):
                    axs[1].add_patch(Rectangle(((f1['descr']['rt_min']), f1['descr']['mz_min']), \
                                               ((f1['descr']['rt_max']) - (f1['descr']['rt_min'])), \
                                               (f1['descr']['mz_max'] - f1['descr']['mz_min']), fc='#C2D0D6', \
                                               linestyle="solid", color='grey', linewidth=2, zorder=0,
                                               in_layout=True,
                                               picker=False))
                    axs[1].annotate(i.split(':')[1], (f1['descr']['rt_max'], f1['descr']['mz_max']),
                                    bbox=dict(facecolor=a_col, alpha=0.3, boxstyle='circle', edgecolor='white',
                                              in_layout=False),
                                    wrap=False, picker=True, fontsize=fsize)
            _vis_feature(fdict=self.feat[ii], id=ii, ax=axs[0])
            # im = axs[1].scatter(f1['fdata']['st'], mz,
            #                     c=np.log(f1['fdata']['I_raw']), s=5, cmap=cm)

        # cbaxes = inset_axes(axs[1], width="30%", height="5%", loc=3)
        # plt.colorbar(im, cax=cbaxes, ticks=[0., 1], orientation='horizontal')

        def p_text(event):
            # print(event.artist)
            ids = str(event.artist.get_text())
            if event.mouseevent.button is MouseButton.LEFT:
                # print('left click')
                # print('id:' + str(ids))
                axs[0].clear()
                axs[0].set_title('')
                _vis_feature(self.feat['id:' + ids], id='id:' + ids, ax=axs[0], add=False)
                event.canvas.draw()
            if event.mouseevent.button is MouseButton.RIGHT:
                # print('right click')
                # print('id:' + str(ids))
                _vis_feature(self.feat['id:' + ids], id='id:' + ids, ax=axs[0], add=True)
                event.canvas.draw()

        cid1 = fig.canvas.mpl_connect('pick_event', p_text)
        axs[1].set_xlabel("Scan time, s")
        axs[1].yaxis.offsetText.set_visible(False)
        axs[1].yaxis.set_label_text("m/z")

    def isotopes(self, selection: Union[dict, None] = None):
        """Visualise feature (peak picked) data.

        Interactive scantime vs m/z plot with features highlighted, m/z and Intensity plot of selected features

        Args:
            selection: M/z and scantime (in sec) window
            lev: Feature level for marking features (`2` or `3`)
        """

        if selection is None:
            selection = self.selection

        # TODO:
        # label feature with id in m/z Int plot
        # two panel plot: bottom m/z over st, top: st vs intentsity upon click on feature
        # basically like vis spectrum but including rectangles and all points but those in fgr feature(s) are s=0.1 and c='grey

        # @log
        def _vis_feature(fdict, id, ax=None, add=False):
            # idx = np.where(self.Xf[:, 6] == fdict['id'])[0]
            if isinstance(ax, type(None)):
                fig = plt.figure()
                ax = fig.add_subplot(111)

            fwhm = self._fwhmBound(id, rtDelta=1) #(ul1, ll1)
            if fwhm is not None:
            # print(fwhm)
                ax.vlines(fwhm[0], 0, np.max(fdict['fdata']['I_smooth']))
                ax.vlines(fwhm[1], 0, np.max(fdict['fdata']['I_smooth']))
            iid=fdict['qc']['icent']
            ax.scatter(fdict['fdata']['st'][iid], fdict['fdata']['I_sm_bline'][iid], c='red')

            ax.plot(fdict['fdata']['st'], fdict['fdata']['I_raw'], label='raw', linewidth=0.5, color='black',
                    zorder=0)

            if fdict['flev'] == 3:
                ax.plot(fdict['fdata']['st'], fdict['fdata']['I_smooth'], label='smoothed', linewidth=0.5,
                        color='cyan', zorder=0)
                # ax.plot(fdict['fdata']['st'], fdict['fdata']['I_bl'], label='baseline', linewidth=0.5,
                #         color='black', ls='dashed', zorder=0)
                ax.plot(fdict['fdata']['st'], fdict['fdata']['I_sm_bline'], label='smoothed bl-cor',
                        color='black', zorder=0)
            if not add:
                ax.set_title(id + f',  flev: {fdict["flev"]}')
                ax.legend()
            else:
                old = axs[0].get_title()
                ax.set_title(old + '\n' + id + f',  flev: {fdict["flev"]}')
            ax.set_ylabel(r"$\bfCount$")
            ax.set_xlabel(r"$\bfScan time$, s")

            data = np.array([list(fdict['qc'].keys()), np.round(list(fdict['qc'].values()), 2)]).T
            column_labels = ["descr", "value"]
            ax.table(cellText=data, colLabels=column_labels, loc='lower right', colWidths=[0.1] * 3)
            return ax

        # which feature - define plot axis boundaries
        Xsub = self._window_mz_rt(self.Xf, selection, allow_none=False, return_idc=False)
        fid_window = np.unique(Xsub[7])
        # l2_ffeat = list(compress(self.feat_l2, np.in1d([float(i.split(':')[1]) for i in self.feat_l2], fid_window)))
        # l3_ffeat = list(compress(self.feat_l3, np.in1d([float(i.split(':')[1]) for i in self.feat_l3], fid_window)))

        # cm = plt.cm.get_cmap('gist_ncar')
        fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 2]}, figsize=(8, 10),
                                constrained_layout=False)
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        axs[0].set_ylabel(r"$\bfCount$")
        axs[0].set_xlabel(r"$\bfScan time$, s")

        axs[1].set_facecolor('#EBFFFF')
        axs[1].text(1.03, 0, self.dpath, rotation=90, fontsize=6, transform=axs[1].transAxes)

        # fill axis for raw data results
        idcA = Xsub[2] > self.noise_thres
        idc_above = np.where(idcA)[0]
        idc_below = np.where(~idcA)[0]

        cm = plt.cm.get_cmap('rainbow')
        axs[1].scatter(Xsub[3, idc_below], Xsub[1, idc_below], s=0.1, c='gray', alpha=0.5)

        imax = np.log(Xsub[2, idc_above])
        psize = (imax / np.max(imax)) * 5
        im = axs[1].scatter(Xsub[3, idc_above], Xsub[1, idc_above], c=Xsub[2, idc_above], s=psize, cmap=cm,
                            norm=LogNorm())

        # Xf
        # 0: scanIdOri,
        # 1: mz,
        # 2: Int,
        # 3: st,
        # 4: scanIdNorm,
        # 5: noiseBool,
        # 6: st_adj,
        # 7: clMem

        # axs[1].scatter(Xsub[Xsub[:, 4] == 0, 3], Xsub[Xsub[:, 4] == 0, 1], s=0.1, c='gray', alpha=0.5)
        cosl= plt.get_cmap('tab20c').colors

        fid = 0
        f = 0

        isot_feat = self.l3Df.id.values
        isot_gr = [int(x.split('_')[0]) if x is not None else None for x in self.l3Df.iPat.values]
        ifg_count = 0
        counter = 0
        p = isot_gr[0]
        for i in isot_feat:

            if p is None:
                a_col ='gray'
            else:
                if p != isot_gr[counter]:
                    tt = (ifg_count +3) // 4
                    ifg_count =  tt*4 if tt <5 else 0
                    a_col = cosl[ifg_count]
                    p=isot_gr[counter]
                    ifg_count += 1
                else:
                    a_col = cosl[ifg_count]
                    ifg_count += 1
                counter += 1

            fid = float(i.split(':')[1])
            f1 = self.feat[i]
            fsize = 6
            mz = Xsub[1, Xsub[7] == fid]
            if len(mz) == len(f1['fdata']['st']):
                # im = axs[1].scatter(f1['fdata']['st'], mz,
                #                     c=np.log(f1['fdata']['I_raw']), s=5, cmap=cm)
                axs[1].add_patch(Rectangle(((f1['descr']['rt_min']), f1['descr']['mz_min']), \
                                           ((f1['descr']['rt_max']) - (f1['descr']['rt_min'])), \
                                           (f1['descr']['mz_max'] - f1['descr']['mz_min']), fc='#C2D0D6', \
                                           linestyle="solid", color='grey', linewidth=2, zorder=0,
                                           in_layout=True,
                                           picker=False))
                axs[1].annotate(i.split(':')[1], (f1['descr']['rt_max'], f1['descr']['mz_max']),
                                bbox=dict(facecolor=a_col, alpha=1, boxstyle='circle', edgecolor='white',
                                          in_layout=False),
                                wrap=False, picker=True, fontsize=fsize)
        id=self.l3Df.id[self.l3Df.smbl.argmax()]
        _vis_feature(fdict=self.feat[id], id=id, ax=axs[0])


        def p_text(event):
            ids = str(event.artist.get_text())
            if event.mouseevent.button is MouseButton.LEFT:
                # print('left click')
                # print('id:' + str(ids))
                axs[0].clear()
                axs[0].set_title('')
                _vis_feature(self.feat['id:' + ids], id='id:' +ids, ax=axs[0], add=False)
                event.canvas.draw()
            if event.mouseevent.button is MouseButton.RIGHT:
                # print('right click')
                # print('id:' + str(ids))
                _vis_feature(self.feat['id:' + ids], id='id:' +ids, ax=axs[0], add=True)
                event.canvas.draw()

        cid1 = fig.canvas.mpl_connect('pick_event', p_text)
        axs[1].set_xlabel(r"$\bfScan time$, s")
        axs[1].yaxis.offsetText.set_visible(False)
        axs[1].yaxis.set_label_text(r"$\bfm/z$")
