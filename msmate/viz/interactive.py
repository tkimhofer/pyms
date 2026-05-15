
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterSciNotation


def _massSpectrumDIA(self, rt: float = 150., scantype: str = '0_2_2_5_20.0', viz: bool = False):
    """Single mass spectrum visualisation (subplot create in ms2L function)."""
    df = self.dfd[scantype]
    idx = np.argmin(np.abs(df.Rt - rt))
    inf = df.iloc[idx]
    sub = self.xrawd[scantype]
    sub = sub[..., sub[0] == (inf.Id - 1)]
    if viz:
        f, ax = plt.subplots(1, 1)
        mz = sub[1]
        intens = sub[2]
        ax.vlines(sub[1], np.zeros_like(mz), intens / np.max(intens) * 100)
        ax.text(0.77, 0.95, f'Acq Type: {inf.LevPP.upper()}', rotation=0, fontsize=8, transform=ax.transAxes,
                ha='left')
        ax.text(0.77, 0.9, f'Rt: {inf.Rt} s', rotation=0, fontsize=8, transform=ax.transAxes, ha='left')
        ax.text(0.77, 0.85, f'Collision E: {inf.Collision_Energy_Act} eV', rotation=0, fontsize=8,
                transform=ax.transAxes, ha='left')
        return (f, ax)
    else:
        return (sub[1], sub[2], np.zeros_like(sub[0]), inf)

def ms2L(self, q_noise, selection, qcm_local: bool = True):
    """Interactive visualisation of ms level 1 and ms level 2 data acquired in data independent scan modes.

        Args:
            q_noise: Quantile probability defining noise intensity threshold (`float`).
            selection: `dict` of scantime (in seconds) and m/z window ranges
            qcm_local: `bool` indicting if local quantile values should be calculated for intensity threshold and colouring

        ### Usage:
        The following example demonstrates the use of the ms2L function:
        ```python
          import msmate as ms
          d = ms.MsExp.bruker(msmate)

          s1 = {'mz_min': 100, 'mz_max': 115, 'rt_min': 60, 'rt_max': 70}
          d.ms2L(q_noise=0.99, selection = s1, qcm_local=True)
          # ms level 2 scan ids can be in/decreased using left/right arrow keys
        ```
    """

    mpl.rcParams['keymap.back'].remove('left') if ('left' in mpl.rcParams['keymap.back']) else None

    def on_lims_change(event_ax: plt.axis):
        x1, x2 = event_ax.get_xlim()
        y1, y2 = event_ax.get_ylim()
        selection = {'mz_min': y1, 'mz_max': y2, 'rt_min': x1, 'rt_max': x2}
        Xs = self._window_mz_rt(self.xrawd[self.ms0string], selection, allow_none=False)
        bsMin = (y2 - y1) / 200 if ((y2 - y1) / 200) > 0.0001 else 0.0001
        bw = bsMin
        y, x = self._get_density(Xs, bw, q_noise=q_noise)
        axs[1, 1].clear()
        axs[1, 1].set_ylim([y1, y2])
        axs[1, 1].plot(x, y, c='black')
        axs[1, 1].tick_params(labelleft=False)
        axs[1, 1].set_xticks([])
        axs[1, 1].annotate(f'bw: {np.round(bw, 5)}\np: {np.round(q_noise, 3)}',
                           xy=(1, y1 + np.min([0.1, float((y2 - y1) / 100)])), fontsize='xx-small', ha='right')
        axs[1, 1].text(1.05, 0, self.fname, rotation=90, fontsize=4, transform=axs[1, 1].transAxes)
        axs[1, 1].tick_params(axis='y', direction='in')

    def on_key(event_ax: plt.axis):
        self.xind.set_visible(False)
        xl = axs[0, 0].get_xlim()
        # xu = axs[0, 0].get_ylim()

        if event_ax.key == 'f':
            self.rtPlot = event_ax.xdata
            axs[0, 0].clear()
            mz, intens, ylim0, inf = self._massSpectrumDIA(rt=self.rtPlot, scantype=self.ms1string, viz=False)

        if event_ax.key == 'right':
            self.rtPlot = self.dfd[self.ms1string].iloc[
                np.argmin(np.abs(self.rtPlot - self.dfd[self.ms1string].Rt)) + 1].Rt
            xl = axs[0, 0].get_xlim()
            axs[0, 0].clear()
            mz, intens, ylim0, inf = self._massSpectrumDIA(rt=self.rtPlot, scantype=self.ms1string, viz=False)
            axs[0, 0].set_xlim(xl)

        if event_ax.key == 'left':
            self.rtPlot = self.dfd[self.ms1string].iloc[
                np.argmin(np.abs(self.rtPlot - self.dfd[self.ms1string].Rt)) - 1].Rt
            xl = axs[0, 0].get_xlim()
            axs[0, 0].clear()
            mz, intens, ylim0, inf = self._massSpectrumDIA(rt=self.rtPlot, scantype=self.ms1string, viz=False)
            axs[0, 0].set_xlim(xl)

        axs[0, 0].vlines(mz, ylim0, intens)
        axs[0, 0].text(0.77, 0.85, f'Acq Type: {inf.LevPP.uppder() if inf.LevPP is not None else "-"}', rotation=0,
                       fontsize='xx-small',
                       transform=axs[0, 0].transAxes, ha='left')
        axs[0, 0].text(0.77, 0.75, f'Rt: {np.round(inf.Rt, 2)} s ({inf.Id})', rotation=0, fontsize='xx-small',
                       transform=axs[0, 0].transAxes, ha='left')
        axs[0, 0].text(0.77, 0.65, f'Collision E: {inf.Collision_Energy_Act} eV', rotation=0, fontsize='xx-small',
                       transform=axs[0, 0].transAxes, ha='left')
        axs[0, 0].text(0.77, 0.55, f'Parent SID: {inf.Parent}', rotation=0, fontsize='xx-small',
                       transform=axs[0, 0].transAxes, ha='left')
        axs[0, 0].text(0.77, 0.45, f'Isol mass: {np.round(inf.MSMS_IsolationMass_Act, 2)}', rotation=0,
                       fontsize='xx-small',
                       transform=axs[0, 0].transAxes, ha='left')
        axs[0, 0].text(0.77, 0.35, f'Q Iso-Res: {np.round(inf.Quadrupole_IsolationResolution_Act, 2)}', rotation=0,
                       fontsize='xx-small', transform=axs[0, 0].transAxes, ha='left')

        d = (axs[1, 0].transData + axs[1, 0].transAxes.inverted()).transform((self.rtPlot, 100))
        self.xind = axs[1, 0].scatter([d[0]], [-0.01], c='red', clip_on=False, marker='^',
                                      transform=axs[1, 0].transAxes)

    Xsub = self._window_mz_rt(self.xrawd[self.ms0string], selection, allow_none=False)
    idc_below, idc_above = self._noiseT(p=q_noise, X=Xsub, local=qcm_local)

    cm = plt.cm.get_cmap('rainbow')
    fig, axs = plt.subplots(2, 2, sharey='row', gridspec_kw={'width_ratios': [3, 1], 'height_ratios': [1.5, 3]})
    axs[1, 1].text(1.05, 0, self.dpath, rotation=90, fontsize=4, transform=axs[1, 1].transAxes)
    axs[1, 1].tick_params(axis='y', direction='in')
    axs[1, 0].scatter(Xsub[3, idc_below], Xsub[1, idc_below], s=0.1, c='gray', alpha=0.5)
    im = axs[1, 0].scatter(Xsub[3, idc_above], Xsub[1, idc_above], c=(Xsub[2, idc_above]), s=5, cmap=cm,
                           norm=LogNorm())
    cbaxes = fig.add_axes([0.15, 0.44, 0.021, 0.1])
    cb = fig.colorbar(mappable=im, cax=cbaxes, orientation='vertical',
                      format=LogFormatterSciNotation(base=10, labelOnlyBase=False))
    cb.ax.tick_params(labelsize='xx-small')
    fig.subplots_adjust(wspace=0.05)
    axs[1, 0].callbacks.connect('ylim_changed', on_lims_change)
    axs[1, 0].callbacks.connect('ylim_changed', on_lims_change)
    cid = fig.canvas.mpl_connect('key_press_event', on_key)

    self.rtPlot = Xsub[3, np.argmax(Xsub[2])]
    mz, intens, ylim0, inf = self._massSpectrumDIA(rt=self.rtPlot, scantype=self.ms1string, viz=False)
    # self.intensmax = np.max(intens)
    axs[0, 0].vlines(mz, ylim0, intens)
    axs[0, 0].text(0.75, 0.85, f'Acq Type: {inf.LevPP.uppder() if inf.LevPP is not None else "-"}', rotation=0,
                   fontsize='xx-small', transform=axs[0, 0].transAxes, ha='left')
    axs[0, 0].text(0.75, 0.75, f'Rt: {inf.Rt} s ({inf.Id})', rotation=0, fontsize='xx-small',
                   transform=axs[0, 0].transAxes, ha='left')
    axs[0, 0].text(0.75, 0.65, f'Collision E: {inf.Collision_Energy_Act} eV', rotation=0, fontsize='xx-small',
                   transform=axs[0, 0].transAxes, ha='left')
    axs[0, 0].text(0.75, 0.55, f'Parent SID: {inf.Parent}', rotation=0, fontsize='xx-small',
                   transform=axs[0, 0].transAxes, ha='left')
    axs[0, 0].text(0.75, 0.45, f'Isol mass: {np.round(inf.MSMS_IsolationMass_Act, 2)}', rotation=0,
                   fontsize='xx-small', transform=axs[0, 0].transAxes, ha='left')
    axs[0, 0].text(0.75, 0.35, f'Q Iso-Res: {np.round(inf.Quadrupole_IsolationResolution_Act, 2)}', rotation=0,
                   fontsize='xx-small', transform=axs[0, 0].transAxes, ha='left')
    axs[0, 0].tick_params(axis='y', direction='out')
    axs[0, 1].set_axis_off()

    axs[0, 0].xaxis.set_label_text(r"$\bfm/z$")
    axs[1, 0].set_xlabel(r"$\bfScan time$ [sec]")
    axs[1, 0].yaxis.set_label_text(r"$\bfm/z$")

    d = (axs[1, 0].transData + axs[1, 0].transAxes.inverted()).transform((self.rtPlot, 100))
    self.xind = axs[1, 0].scatter([d[0]], [-0.01], c='red', clip_on=False, marker='^',
                                  transform=axs[1, 0].transAxes)
    fig.subplots_adjust(wspace=0.05, hspace=0.3)



    @staticmethod
    def _get_density(X: np.ndarray, bw: float, q_noise: float = 0.5):
        """Calculation of m/z dimension kernel density"""
        xmin = X[1].min()
        xmax = X[1].max()
        b = np.linspace(xmin, xmax, 500)
        idxf = np.where(X[2] > np.quantile(X[2], q_noise))[0]
        x_rev = X[1, idxf]
        kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(x_rev[:, np.newaxis])
        log_dens = np.exp(kde.score_samples(b[:, np.newaxis]))
        return (b, log_dens / np.max(log_dens))

    def viz_mz_rt_density(
            self,
            q_noise: float = 0.50,
            selection: ScanWindow = None,
            qcm_local: bool = True,
    ):
        """Interactive RT vs m/z plot with updating m/z density panel."""

        import matplotlib.patches as patches
        from matplotlib.colors import LogNorm
        from matplotlib.ticker import LogFormatterSciNotation

        if selection is None:
            selection = ScanWindow()

        fname_fontsize = 7

        Xraw = self.xrawd[self.ms0string]

        def update_density(event_ax):
            nonlocal updating

            if updating:
                return

            updating = True
            try:
                x1, x2 = axs[0].get_xlim()
                y1, y2 = axs[0].get_ylim()

                sel = ScanWindow(
                    mz_min=min(y1, y2),
                    mz_max=max(y1, y2),
                    st_min=min(x1, x2),
                    st_max=max(x1, x2),
                )

                Xs = self._window_mz_rt(Xraw, selection=sel)

                axs[1].clear()
                axs[1].set_ylim([y1, y2])
                axs[1].tick_params(labelleft=False)
                axs[1].set_xticks([])
                axs[1].tick_params(axis="y", direction="in")

                axs[1].text(
                    1.05, 0.02,
                    str(self.fname),
                    rotation=90,
                    fontsize=fname_fontsize,
                    transform=axs[1].transAxes,
                    va="bottom",
                    bbox=dict(
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.75,
                        pad=1.0,
                    ),
                )

                if Xs.shape[1] > 0:
                    bw = max((y2 - y1) / 200, 0.0001)
                    y, x = self._get_density(Xs, bw, q_noise=q_noise)

                    axs[1].plot(x, y, c="black", linewidth=0.8)

                    axs[1].text(
                        0.96, 0.96,
                        f"bw: {bw:.5f}\np: {q_noise:.3f}",
                        transform=axs[1].transAxes,
                        fontsize="xx-small",
                        ha="right",
                        va="top",
                        linespacing=1.15,
                        bbox=dict(
                            facecolor="white",
                            edgecolor="none",
                            alpha=0.85,
                            pad=1.5,
                        ),
                    )

                fig.canvas.draw_idle()

            finally:
                updating = False

        Xsub = self._window_mz_rt(Xraw, selection=selection)

        idc_below, idc_above = self._noiseT(
            p=q_noise,
            X=Xsub,
            local=qcm_local,
        )

        fig, axs = plt.subplots(
            1, 2,
            sharey=False,
            gridspec_kw={"width_ratios": [3, 1]},
        )

        axs[1].tick_params(labelleft=False)
        axs[1].set_xticks([])
        axs[1].tick_params(axis="y", direction="in")

        axs[1].text(
            1.05, 0.02,
            str(self.fname),
            rotation=90,
            fontsize=fname_fontsize,
            transform=axs[1].transAxes,
            va="bottom",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.75,
                pad=1.0,
            ),
        )

        if len(idc_below) > 0:
            axs[0].scatter(
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
                im = axs[0].scatter(
                    Xsub[IDX_ST, idc_above][pos],
                    Xsub[IDX_MZ, idc_above][pos],
                    c=vals[pos],
                    s=5,
                    cmap=plt.colormaps["viridis"],
                    norm=LogNorm(),
                )

                # white underlay covering colorbar AND tick labels
                bbox = axs[0].get_position()

                cb_box = [
                    bbox.x0 + 0.03 * bbox.width,
                    bbox.y0 + 0.84 * bbox.height,
                    0.025 * bbox.width,
                    0.12 * bbox.height,
                ]

                bg_box = [
                    cb_box[0] - 0.006,
                    cb_box[1] - 0.008,
                    cb_box[2] + 0.055,
                    cb_box[3] + 0.018,
                ]

                bg = patches.Rectangle(
                    (bg_box[0], bg_box[1]),
                    bg_box[2],
                    bg_box[3],
                    transform=fig.transFigure,
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.90,
                    zorder=2,
                )
                fig.patches.append(bg)

                cbaxes = fig.add_axes(cb_box, zorder=3)
                cbaxes.set_facecolor("white")
                cbaxes.patch.set_alpha(1.0)

                cb = fig.colorbar(
                    mappable=im,
                    cax=cbaxes,
                    orientation="vertical",
                    format=LogFormatterSciNotation(
                        base=10,
                        labelOnlyBase=False,
                    ),
                )

                cb.ax.set_facecolor("white")
                cb.ax.tick_params(labelsize="xx-small", length=2)

                for lab in cb.ax.get_yticklabels():
                    lab.set_bbox(dict(
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.90,
                        pad=0.5,
                    ))

        axs[0].set_xlabel("Scan time [sec]")
        axs[0].set_ylabel("m/z")

        updating = False
        axs[0].callbacks.connect("ylim_changed", update_density)
        axs[0].callbacks.connect("xlim_changed", update_density)

        update_density(axs[0])

        fig.subplots_adjust(wspace=0.05)

        return fig, axs
