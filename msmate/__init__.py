




class ms_exp:
    # import methods
    from ._peak_picking import est_intensities, cl_summary, feat_summary, peak_picking
    from ._reading import read_mzml, read_bin, exp_meta, node_attr_recurse, plt_chromatograms, rec_attrib, extract_mass_spectrum_lev1

    def __init__(self, path):

        self.path=path
        import subprocess
        cmd = 'find ' + self.path + ' -type f -iname *' + 'mzML'
        sp = subprocess.getoutput(cmd)
        self.files = sp.split('\n')
        self.nfiles=len(self.files)
        if self.nfiles>0:
            print('Number of mzML files in directory: ' + str(self.nfiles))
            print('First one:' + str(self.files[0]))
        else:
            print('No mzML files found in directory - check path variable.')

    def peak(self, st_len_min=10, plot=True, mz_bound=0.001, rt_bound=3):
        self.st_len_min=st_len_min
        self.peak_picking(self.X2, self.st_len_min, plot, mz_bound, rt_bound)

    def read_exp(self, fidx, mslevel='1', print_summary=True):
        import numpy as np
        if fidx > (len(self.files)-1): print('Index too high (max value is ' + str(self.nfiles-1) + ').')
        self.fidx=fidx
        self.flag=str(mslevel)

        out = self.read_mzml(sidx=self.fidx, flag=self.flag, print_summary=print_summary)

        print(out[0])
        if 'srm' in out[0]:
            self.dstype, self.df, self.scantime, self.I, self.summary = out

        if 'untargeted' in out[0]:
            self.dstype, self.df, self.scanid, self.mz, self.I, self.scantime, self.summary = out
            # out = self.read_mzml(path=path, sidx=self.fidx, flag=self.flag, print_summary=print_summary)
            self.nd = len(self.I)
            self.Xraw = np.array([self.scanid, self.mz, self.I, self.scantime]).T

    def window_mz_rt(self, Xr, selection={'mz_min': None, 'mz_max': None, 'rt_min': None, 'rt_max': None},
                     allow_none=False, return_idc=False):
        # make selection on Xr based on rt and maz range
        # allow_none: true if None alowed in selection (then full range is used)

        import numpy as np
        part_none = any([x == None for x in selection.values()])
        all_fill = any([x != None for x in selection.values()])

        if not allow_none:
            if not all_fill:
                raise ValueError('Define plot boundaries with selection argument.')
        if part_none:
            if selection['mz_min'] is not None:
                t1 = Xr[..., 1] >= selection['mz_min']
            else:
                t1 = np.ones(Xr.shape[0], dtype=bool)

            if selection['mz_max'] is not None:
                t2 = Xr[..., 1] <= selection['mz_max']
            else:
                t2 = np.ones(Xr.shape[0], dtype=bool)

            if selection['rt_min'] is not None:
                t3 = Xr[..., 3] > selection['rt_min']
            else:
                t3 = np.ones(Xr.shape[0], dtype=bool)

            if selection['rt_max'] is not None:
                t4 = Xr[..., 3] < selection['rt_max']
            else:
                t4 = np.ones(Xr.shape[0], dtype=bool)
        else:
            t1 = Xr[..., 1] >= selection['mz_min']
            t2 = Xr[..., 1] <= selection['mz_max']
            t3 = Xr[..., 3] > selection['rt_min']
            t4 = Xr[..., 3] < selection['rt_max']

        idx = np.where((t1) & (t2) & (t3) & (t4))[0]

        if return_idc:
            return idx
        else:
            return Xr[idx, ...]

    def vis_spectrum(self, q_noise=0.89, selection={'mz_min': None, 'mz_max': None, 'rt_min': None, 'rt_max': None}):
        # visualise part of the spectrum, before peak picking

        import matplotlib.pyplot as plt
        import numpy as np

        # select Xraw
        Xsub = self.window_mz_rt(self.Xraw, selection, allow_none=False)

        # filter noise points
        noise_thres = np.quantile(self.Xraw[..., 2], q=q_noise)
        idc_above = np.where(Xsub[:, 2] > noise_thres)[0]
        idc_below = np.where(Xsub[:, 2] <= noise_thres)[0]

        cm = plt.cm.get_cmap('rainbow')
        # fig, ax = plt.subplots()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.text(1.25, 0, self.df.fname[0], rotation=90, fontsize=6, transform=ax.transAxes)

        ax.scatter(Xsub[idc_below, 3], Xsub[idc_below, 1], s=0.1, c='gray', alpha=0.5)
        im = ax.scatter(Xsub[idc_above, 3], Xsub[idc_above, 1], c=np.log(Xsub[idc_above, 2]), s=5, cmap=cm)
        fig.canvas.draw()

        ax.set_xlabel(r"$\bfScan time$, s")
        ax2 = ax.twiny()

        ax.yaxis.offsetText.set_visible(False)
        ax.yaxis.set_label_text(r"$\bfm/z$")

        def tick_conv(X):
            V = X / 60
            return ["%.2f" % z for z in V]

        tmax = np.max(Xsub[:, 3])
        tmin = np.min(Xsub[:, 3])

        tick_loc = np.linspace(tmin / 60, tmax / 60, 5) * 60
        # tick_loc = np.arange(np.round((tmax - tmin) / 60, 0), 0.1)*60
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(tick_loc)
        ax2.set_xticklabels(tick_conv(tick_loc))
        ax2.set_xlabel(r"min")
        fig.colorbar(im, ax=ax)
        fig.show()

        return (fig, ax)

    def vis_features(self, selection={'mz_min': 600, 'mz_max': 660, 'rt_min': 600, 'rt_max': 900}, rt_bound=0.51,
                     mz_bound=0.001):
        # visualise mz/rt plot after peak picking
        # basically like vis spectrum but including rectangles and all points but those in fgr feature(s) are s=0.1 and c='grey
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        import numpy as np
        import pandas as pd

        # which feature - define plot axis boundaries
        Xsub = self.window_mz_rt(self.Xf, selection, allow_none=False, return_idc=False)

        labs = np.unique(Xsub[:,5])
        # define L2 clusters
        cl_l2 = self.fs[np.isin(self.fs.cl_id.values, labs)]

        # define points exceeding noise and in L2 feature list
        cl_n_above = np.where((Xsub[:, 4] == 0) & (np.isin(Xsub[..., 5], cl_l2.cl_id.values)))[0]
        cl_n_below = np.where((Xsub[:, 4] == 1) | (~np.isin(Xsub[..., 5], cl_l2.cl_id.values)))[0]
        assert ((len(cl_n_above) + len(cl_n_below)) == Xsub.shape[0])

        cm = plt.cm.get_cmap('gist_ncar')
        # fig, ax = plt.subplots()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.text(1.25, 0, self.df.fname[0], rotation=90, fontsize=6, transform=ax.transAxes)
        ax.scatter(Xsub[cl_n_below, 3], Xsub[cl_n_below, 1], s=0.1, c='gray', alpha=0.1)
        for i in range(cl_l2.shape[0]):
            f1 = cl_l2.iloc[i]
            ax.add_patch(Rectangle(((f1['rt_min'] - rt_bound), f1['mz_min'] - mz_bound), \
                                   ((f1['rt_max']) - (f1['rt_min'])) + rt_bound * 2, \
                                   (f1['mz_max'] - f1['mz_min']) + mz_bound * 2, fc='gray', alpha=0.5, \
                                   linestyle="dotted"))

        im = ax.scatter(Xsub[cl_n_above, 3], Xsub[cl_n_above, 1], c=np.log(Xsub[cl_n_above, 2]), s=5, cmap=cm)
        fig.canvas.draw()
        ax.set_xlabel(r"$\bfScan time$, s")
        ax2 = ax.twiny()

        ax.yaxis.offsetText.set_visible(False)
        # ax1.yaxis.set_label_text("original label" + " " + offset)
        ax.yaxis.set_label_text(r"$\bfm/z$" )

        def tick_conv(X):
            V = X / 60
            return ["%.2f" % z for z in V]
        tmax=np.max(Xsub[:,3])
        tmin=np.min(Xsub[:,3])

        tick_loc = np.linspace(tmin/60, tmax/60, 5) * 60
        # tick_loc = np.arange(np.round((tmax - tmin) / 60, 0), 0.1)*60
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(tick_loc)
        ax2.set_xticklabels(tick_conv(tick_loc))
        ax2.set_xlabel(r"min")
        fig.colorbar(im, ax=ax)

        return (fig, ax)

    def find_isotopes(self):
        import numpy as np
        # deconvolution based on isotopic mass differences (1 m/z)
        # check which features overlap in rt dimension
        S = np.arange(self.fs.shape[0]).tolist()
        isot = []
        while len(S) > 0:
            i = S[0]
            #  ------
            #    ---y--
            # --x--
            x_ix = ((self.fs.rt_min.values >= self.fs.iloc[i].rt_min) & (
                        self.fs.rt_min.values <= self.fs.iloc[i].rt_max))
            y_ix = ((self.fs.rt_max.values >= self.fs.iloc[i].rt_min) & (
                        self.fs.rt_max.values <= self.fs.iloc[i].rt_max))
            idx = np.where(x_ix | y_ix)[0]

            # rowselection, idx dscr pl1
            pl2 = self.fs.iloc[idx]
            mzdiff = np.round(pl2.mz_mean.values - self.fs.iloc[i].mz_mean)
            idx_im = np.where((mzdiff < 5) & (mzdiff >= -2))[0]

            if len(idx_im) > 1:
                if len(idx_im) > 3: print(i)
                ints = pl2.int_integral.iloc[idx_im] / np.max(pl2.int_integral.iloc[idx_im])
                imax = np.where(ints == 1)[0]
                icent = np.where((ints > 0.2) & (ints < 1))[0]
                ilow = np.where((ints > 0.08) & (ints < 0.2))[0]
                ilow1 = np.where((ints > 0) & (ints < 0.08))[0]

                if len(icent) > 1 | len(ilow) > 1:
                    print(i)
                    raise ValueError('check')

                ik = np.concatenate([imax, icent, ilow, ilow1])
                ifound = {'seed': i, \
                          'ridx': idx[idx_im[ik]], \
                          'relI': ints.iloc[ik].values}

                isot.append(ifound)
                [S.remove(item) for item in ifound['ridx'] if item in S]
            else:
                S.remove(i)

        self.isotopes=isot

    def vis_ipattern(self, fgr, b_mz=10, b_rt=10, rt_bound=0.51, mz_bound=0.2):
        # visualise mz/rt plot after peak picking and isotope detection
        # basically like vis spectrum but including rectangles and all points but those in fgr feature(s) are s=0.1 and c='grey
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        import numpy as np

        # which feature - define plot axis boundaries
        mz_min = self.fs.mz_min.iloc[fgr['ridx']].min() - b_mz / 2
        mz_max = self.fs.mz_max.iloc[fgr['ridx']].max() + b_mz / 2

        rt_min = self.fs.rt_min.iloc[fgr['ridx']].min() - b_rt / 2
        rt_max = self.fs.rt_max.iloc[fgr['ridx']].max() + b_rt / 2

        Xsub = self.window_mz_rt(self.Xf,
                                 selection={'mz_min': mz_min, 'mz_max': mz_max, 'rt_min': rt_min, 'rt_max': rt_max},
                                 allow_none=False, return_idc=False)

        # define L2 clusters
        cl_l2 = self.fs.iloc[fgr['ridx']]

        # define points exceeding noise and in L2 feature list
        cl_n_above = np.where((Xsub[:, 4] == 0) & (np.isin(Xsub[..., 5], cl_l2.cl_id.values)))[0]
        cl_n_below = np.where((Xsub[:, 4] == 1) | (~np.isin(Xsub[..., 5], cl_l2.cl_id.values)))[0]
        assert ((len(cl_n_above) + len(cl_n_below)) == Xsub.shape[0])

        cm = plt.cm.get_cmap('gist_ncar')
        # fig, ax = plt.subplots()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.text(1.25, 0, self.df.fname[0], rotation=90, fontsize=6, transform=ax.transAxes)
        ax.scatter(Xsub[cl_n_below, 3], Xsub[cl_n_below, 1], s=0.5, c='gray', alpha=0.7)
        for i in range(cl_l2.shape[0]):
            f1 = cl_l2.iloc[i]
            ax.add_patch(Rectangle(((f1['rt_min'] - rt_bound), f1['mz_min'] - mz_bound), \
                                   ((f1['rt_max']) - (f1['rt_min'])) + rt_bound * 2, \
                                   (f1['mz_max'] - f1['mz_min']) + mz_bound * 2, fc='gray', alpha=0.5, \
                                   linestyle="dotted"))

        im = ax.scatter(Xsub[cl_n_above, 3], Xsub[cl_n_above, 1], c=np.log(Xsub[cl_n_above, 2]), s=5, cmap=cm)
        fig.canvas.draw()
        ax.set_xlabel(r"$\bfScan time$, s")
        ax2 = ax.twiny()

        ax.yaxis.offsetText.set_visible(False)
        # ax1.yaxis.set_label_text("original label" + " " + offset)
        ax.yaxis.set_label_text(r"$\bfm/z$")

        def tick_conv(X):
            V = X / 60
            return ["%.2f" % z for z in V]

        tmax = np.max(Xsub[:, 3])
        tmin = np.min(Xsub[:, 3])

        tick_loc = np.linspace(tmin / 60, tmax / 60, 5) * 60
        # tick_loc = np.arange(np.round((tmax - tmin) / 60, 0), 0.1)*60
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(tick_loc)
        ax2.set_xticklabels(tick_conv(tick_loc))
        ax2.set_xlabel(r"min")
        fig.colorbar(im, ax=ax)

        return (fig, ax)