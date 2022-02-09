class list_exp:
    def __init__(self, path, ftype='mzml', nmax=None, print_summary=True):

        import pandas as pd
        import numpy as np

        self.ftype = ftype
        self.path = path
        self.files = []
        self.nmax=nmax

        import glob, os
        self.path=os.path.join(path, '')

        c = 1
        for file in glob.glob(self.path+"*."+ftype):
            self.files.append(file)
            if c == nmax: break
            c += 1


        # cmd = 'find ' + self.path + ' -type f -iname *' + ftype
        # sp = subprocess.getoutput(cmd)
        # self.files = sp.split('\n')
        self.nfiles = len(self.files)

        # check if bruker fits and qc's are included
        fsize = list()
        mtime = list()
        # check if procs exists
        for i in range(self.nfiles):
            fname = self.files[i]
            inf = os.stat(fname)
            try:
                inf
            except NameError:
                mtime.append(None)
                fsize.append(None)
                continue

            mtime.append(inf.st_mtime)
            fsize.append(inf.st_size)

        self.df = pd.DataFrame({'exp':self.files, 'fsize_mb': fsize, 'mtime': mtime})
        self.df['fsize_mb'] = fsize
        self.df['fsize_mb'] = self.df['fsize_mb'].values / 1000 ** 2
        self.df['mtime'] = mtime
        mtime_max, mtime_min = (np.argmax(self.df['mtime'].values), np.argmin(self.df['mtime'].values))
        fsize_mb_max, fsize_mb_min = (np.argmax(self.df['fsize_mb'].values), np.argmin(self.df['fsize_mb'].values))
        self.df['mtime'] = pd.to_datetime(self.df['mtime'], unit='s').dt.floor('T')

        # self.df.exp.str.split('/')

        self.df.exp = [os.path.relpath(self.df.exp.values[x], path) for x in range(len(self.df.exp.values))]
        self.files =  self.df.exp.values
        # os.path.relpath(self.df.exp.values[0], path)

        self.df.index = ['s' + str(i) for i in range(self.df.shape[0])]

        self.n_exp = (self.df.shape[0])

        print('---')
        print(f'Number of mzml files: {self.n_exp}')
        print(f'Files created from { self.df.mtime.iloc[mtime_min]} to { self.df.mtime.iloc[mtime_max]}')
        print(f'Files size ranging from {round(self.df.fsize_mb.iloc[fsize_mb_min],1)} MB to {round(self.df.fsize_mb.iloc[fsize_mb_max], 1)} MB')
        print(f'For detailed experiment list see obj.df')
        print('---')
        # summary = self.df.groupby(['exp']).agg(n=('fsize'), size_byte=('fsize', 'mean')),
        #                                   maxdiff_byte=('size', lambda x: max(x) - min(x)),
        #                                   mtime=('mtime', 'max')).reset_index()
        # summary.sort_values(by='n', ascending=False)
        # summary.mtime = pd.to_datetime(summary.mtime, unit='s').dt.floor('T')

        # if print_summary:
        #     print(summary)
        #
        # if self.nfiles > 0:
        #     print('Number of mzML files found: ' + str(self.nfiles))
        #     print('First one:' + str(self.files[0]))
        # else:
        #     print('No mzML files found in directory - check path variable.')
        #

class ExpSet(list_exp):
    from ._btree import bst_search, spec_distance

    def __init__(self, path,  msExp, ftype='mzml', nmax=None):
        super().__init__(path, ftype=ftype, print_summary=True, nmax=nmax)

    def read(self, pattern=None, multicore=True):
        import os
        import numpy as np
        if not isinstance(pattern, type(None)):
            import re
            self.files = list(filter(lambda x: re.search(pattern, x), self.files))
            print('Selected are ' + str(len(self.files)) + ' files. Start importing...')
            self.df = self.df[self.df.exp.isin(self.files)]
            self.df.index = ['s' + str(i) for i in range(self.df.shape[0])]
        else: print('Start importing...')

        if multicore & (len(self.files) > 1):
            import multiprocessing
            from joblib import Parallel, delayed

            ncore = np.min([multiprocessing.cpu_count() - 2, len(self.files)])
            print(f'Using {ncore} workers.')
            self.exp = Parallel(n_jobs=ncore)(delayed(msExp)(os.path.join(self.path, i), '1', False) for i in self.files)
        else:
            # if (len(self.files) > 1):
            #     print('Importing file')
            # else:
            #     print('Importing file')
            self.exp = []
            for f in self.files:
                fpath=os.path.join(self.path, f)
                self.exp.append(msExp(fpath, '1', False))

    def get_bpc(self, plot):
        import numpy as np
        from scipy.interpolate import interp1d

        st = []
        st_delta = []
        I = []
        st_inter = []
        for exp in self.exp:
            df_l1 = exp.df.loc[exp.df.ms_level == '1']
            exp_st = df_l1.scan_start_time
            exp_I = df_l1.base_peak_intensity.astype(float)
            I.append(exp_I)
            st_delta.append(np.median(np.diff(exp_st)))
            st.append(df_l1.scan_start_time)
            st_inter.append(interp1d(exp_st, exp_I, bounds_error=False, fill_value=0))

        tmin = np.concatenate(st).min()
        tmax = np.concatenate(st).max()
        x_new = np.arange(tmin, tmax, np.mean(st_delta))

        # interpolate to common x points
        self.bpc= np.zeros((len(self.exp), len(x_new)))
        self.bpc_st = x_new
        for i in range(len(st_inter)):
            self.bpc[i] = st_inter[i](x_new)

        if plot:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = plt.subplot(111)
            ax.plot(x_new, self.bpc.T, linewidth=0.8)
            #ax.legend(self.df.index.values)

            fig.canvas.draw()
            offset = ax.yaxis.get_major_formatter().get_offset()

            ax.set_xlabel(r"$\bfs$")
            ax.set_ylabel(r"$\bfSum Max Count$")
            ax2 = ax.twiny()

            ax.yaxis.offsetText.set_visible(False)
            # ax1.yaxis.set_label_text("original label" + " " + offset)
            if offset != '': ax.yaxis.set_label_text(r"$\bfInt/count$ ($x 10^" + offset.replace('1e', '') + '$)')

            def tick_conv(X):
                V = X / 60
                return ["%.2f" % z for z in V]

            tick_loc = np.arange(np.round(tmin / 60), np.round(tmax / 60), 2, dtype=float) * 60
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks(tick_loc)
            ax2.set_xticklabels(tick_conv(tick_loc))
            ax2.set_xlabel(r"$\bfmin$")
            # ax2.yaxis.set_label_text(r"$Int/count$ " + offset)

            ax.legend(self.df.index.values, loc='best', bbox_to_anchor=(0.5, 0.5, 0.5, 0.5))

    def get_tic(self, plot):
        import numpy as np
        from scipy.interpolate import interp1d

        st = []
        st_delta = []
        I = []
        st_inter = []
        for exp in self.exp:
            df_l1 = exp.df.loc[exp.df.ms_level == '1']
            exp_st = df_l1.scan_start_time
            exp_I = df_l1.total_ion_current.astype(float)
            I.append(exp_I)
            st_delta.append(np.median(np.diff(exp_st)))
            st.append(df_l1.scan_start_time)
            st_inter.append(interp1d(exp_st, exp_I, bounds_error=False, fill_value=0))

        tmin = np.concatenate(st).min()
        tmax = np.concatenate(st).max()
        x_new = np.arange(tmin, tmax, np.mean(st_delta))

        # interpolate to common x points
        self.tic= np.zeros((len(self.exp), len(x_new)))
        self.tic_st = x_new
        for i in range(len(st_inter)):
            self.tic[i] = st_inter[i](x_new)

        if plot:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = plt.subplot(111)
            ax.plot(x_new, self.tic.T, linewidth=0.8)
            #ax.legend(self.df.index.values)

            fig.canvas.draw()
            offset = ax.yaxis.get_major_formatter().get_offset()

            ax.set_xlabel(r"$\bfs$")
            #ax.set_ylabel(r"$\bfTotal Count$")
            ax2 = ax.twiny()

            ax.yaxis.offsetText.set_visible(False)
            # ax1.yaxis.set_label_text("original label" + " " + offset)
            if offset != '': ax.yaxis.set_label_text(r"$\bfTotal$ $\bfInt/count$ ($x 10^" + offset.replace('1e', '') + '$)')

            def tick_conv(X):
                V = X / 60
                return ["%.2f" % z for z in V]

            tick_loc = np.arange(np.round(tmin / 60), np.round(tmax / 60), 2, dtype=float) * 60
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks(tick_loc)
            ax2.set_xticklabels(tick_conv(tick_loc))
            ax2.set_xlabel(r"$\bfmin$")
            # ax2.yaxis.set_label_text(r"$Int/count$ " + offset)

            ax.legend(self.df.index.values, loc='best', bbox_to_anchor=(0.5, 0.5, 0.5, 0.5))

    def chrom_btree(self, ctype='bpc', index=0, depth=4, minle=100, linkage='ward', xlab='Scantime (s)', ylab='Total Count', ax=None, colour='green'):
        if ctype == 'bpc':
            if not hasattr(self, ctype):
                self.get_bpc(plot=False)
            mm = self.bpc[index]
            mx = self.bpc_st
        elif ctype == 'tic':
            if not hasattr(self, ctype):
                self.get_tic(plot=False)
            mm = self.tic[index]
            mx = self.tic_st

        self.btree = self.bst_search(mx, mm,depth, minle)
        return self.btree.vis_graph(xlab=xlab, ylab=ylab, ax=ax, col=colour)

    def chrom_dist(self, ctype='bpc', depth=4, minle=3, linkage='ward', **kwargs):


        if ctype == 'bpc':
            if not hasattr(self, ctype):
                self.get_bpc(plot=False)
            mm = self.bpc
            mx = self.bpc_st
        elif ctype == 'tic':
            if not hasattr(self, ctype):
                self.get_tic(plot=False)
            mm = self.tic
            mx = self.tic_st

        self.spec_distance(mm, mx, depth, minle, linkage, **kwargs)

    def pick_peaks(self, st_adj=6e2, q_noise=0.89, dbs_par={'eps': 0.02, 'min_samples': 2},
                 selection={'mz_min': None, 'mz_max': None, 'rt_min': None, 'rt_max': None}, multicore=True):
        import numpy as np
        print('Start peak picking...')

        def pp(ex, st_adj, q_noise, dbs_par, selection):
            try:
                ex.peak_picking(st_adj, q_noise, dbs_par, selection, False)
                return ex
            except Exception as e:
                return ex

        self.pp_selection = selection

        if multicore & (len(self.exp) > 1):
            import multiprocessing
            from joblib import Parallel, delayed
            import numpy as np

            ncore = np.min([multiprocessing.cpu_count() - 2, len(self.files)])
            print(f'Using {ncore} workers.')
            self.exp = Parallel(n_jobs=ncore)(delayed(pp)(i, st_adj, q_noise, dbs_par, selection) for i in self.exp)
        else:
            for i in range(len(self.exp)):
                ex=self.exp[i]
                print(ex.fpath)
                self.exp[i]=pp(ex, st_adj, q_noise, dbs_par, selection)

    def vis_peakGroups(self):
        import matplotlib.pyplot as plt

        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        c = 0
        d = len(colors)-1
        for e in range(len(self.exp)):
            ex = self.exp[e]
            if hasattr(ex, 'feat_l2'):
                print(e)
                fl2 = ex.feat_l2
                rt=[]
                mz=[]
                for i in fl2:
                    rt.append(ex.feat[i]['rt_maxI'])
                    mz.append(ex.feat[i]['mz_maxI'])
                plt.scatter(rt, mz, facecolors='none', edgecolors=colors[c], label=self.df.index[c])

                if c == d:
                    c=0
                else:
                    c=c+1
        plt.legend()
        plt.xlabel('Scan time (s)')
        plt.ylabel('m/z')






    # def rt_ali(self, method='warping', parallel=True):
    #     # import sys
    #     # if ((sys.version_info.major > 3) | (sys.version_info.major == 3 $ sys.version_info.minor <8)):
    #     #     raise ValueError('Update Python to 3.9 or higher.')
    #     #
    #
    #     if not hasattr(self, 'bpc'):
    #         self.get_bpc(plot=False)
    #
    #     from fastdtw import fastdtw
    #     from scipy.spatial.distance import euclidean
    #     import numpy as np
    #     import matplotlib.pyplot as plt
    #
    #     self.bpc.shape
    #     self.bpc_st.shape
    #
    #     x =  self.bpc[0]
    #     y = np.mean(self.bpc, 0)
    #     distance, path = fastdtw(x, y, dist=euclidean)
    #     len(path)
    #     np.arange(0, len(x))
    #
    #     x_new = []
    #     y_new = []
    #     ix_last=-1
    #     iy_last=-1
    #     for i in range(len(path)):
    #         x_new.append(x[path[i][0]])
    #         y_new.append(y[path[i][0]])
    #
    #     plt.plot(x_new)
    #     plt.plot(y_new)
    #
    #     plt.plot(x)
    #     plt.plot(y)
    #
    #
    #         if i > 0 :
    #             ix_last = path[i-1][0]
    #             iy_last = path[i - 1][1]
    #
    #         if path[i][0] == ix_last: x_new.append(x[path[i][0]]
    #         else: x_new.append(path[i][0])
    #
    #         if path[i][1] == iy_last: continue
    #         else: y_new.append(path[i][1])
    #
    #
    #     # get warping functions





class msExp:
    # import methods
    from ._peak_picking import est_intensities, cl_summary, feat_summary, peak_picking, vis_feature_pp
    from ._reading import read_mzml, read_bin, exp_meta, node_attr_recurse, rec_attrib, extract_mass_spectrum_lev1

    def __init__(self, fpath, mslev='1', print_summary=False):
        self.fpath = fpath
        self.read_exp(mslevel=mslev, print_summary=print_summary)

    def peak(self, st_len_min=10, plot=True, mz_bound=0.001, rt_bound=3):
        self.st_len_min=st_len_min
        self.peak_picking(self.X2, self.st_len_min, plot, mz_bound, rt_bound)

    def read_exp(self, mslevel='1', print_summary=True):
        import numpy as np

        #if fidx > (len(self.files)-1): print('Index too high (max value is ' + str(self.nfiles-1) + ').')
        self.flag=str(mslevel)

        out = self.read_mzml(flag=self.flag, print_summary=print_summary)
        print('file: ' + self.fpath + '-->' + out[0])

        if 'srm' in out[0]:
            self.dstype, self.df, self.scantime, self.I, self.summary = out

        if 'untargeted' in out[0]:
            self.dstype, self.df, self.scanid, self.mz, self.I, self.scantime, self.summary = out
            # out = self.read_mzml(path=path, sidx=self.fidx, flag=self.flag, print_summary=print_summary)
            self.nd = len(self.I)
            self.Xraw = np.array([self.scanid, self.mz, self.I, self.scantime]).T
        self.stime_conv()

    def d_ppm(self, mz, ppm):
        d = (ppm * mz)/ 1e6
        return mz-(d/2), mz+(d/2)

    def xic(self, mz, ppm, rt_min=None, rt_max=None):
        import numpy as np
        mz_min, mz_max = self.d_ppm(mz, ppm)

        idx_mz = np.where((self.Xraw[:, 1] >= mz_min) & (self.Xraw[:, 1] <= mz_max))[0]
        if len(idx_mz) < 0:
            raise ValueError('mz range not found')

        X_mz = self.Xraw[idx_mz, :]

        sid = np.array(np.unique(X_mz[:, 0]).astype(int))
        xic = np.zeros(int(np.max(self.Xraw[:, 0])+1))
        for i in sid:
            xic[i-1] = np.sum(X_mz[np.where(X_mz[:,0] == i), 2])
        stime = np.sort(np.unique(self.Xraw[:,3]))

        if (~isinstance(rt_min, type(None)) | ~isinstance(rt_max, type(None))):
            idx_rt = np.where((stime >= rt_min) & (stime <= rt_max))[0]
            if len(idx_rt) < 0:
                raise ValueError('rt range not found')
            stime = stime[idx_rt]
            xic = xic[idx_rt]

        return (stime, xic)


    @staticmethod
    def window_mz_rt(Xr, selection={'mz_min': None, 'mz_max': None, 'rt_min': None, 'rt_max': None},
                     allow_none=False, return_idc=False):
        # make selection on Xr based on rt and mz range
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

        idx = np.where(t1 & t2 & t3 & t4)[0]

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

    def stime_conv(self):
        if 'minute' in self.df:
            self.df['scan_start_time']=self.df['scan_start_time'].astype(float)*60
        else:
            self.df['scan_start_time'] = self.df['scan_start_time'].astype(float)

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

    def plt_chromatogram(self, tmin=None, tmax=None, ctype=['tic', 'bpc', 'xic'], xic_mz=[], xic_ppm=10):
        import numpy as np
        import matplotlib.pyplot as plt

        df_l1 = self.df.loc[self.df.ms_level == '1']
        df_l2 = self.df.loc[self.df.ms_level == '2']
        if df_l1.shape[0] == 0: raise ValueError('mslevel 1 in df does not exist or is not defined')

        xic = False
        tic = False
        bpc = False

        if not isinstance(ctype, list):
            ctype=[ctype]
        if any([x in 'xic' for x in ctype]):
            xic = True
        if any([x in 'bpc' for x in ctype]):
            bpc = True
        if any([x in 'tic' for x in ctype]):
            tic = True

        if ((xic == False) & (tic == False) & (bpc == False)):
            raise ValueError('ctype error: define chromatogram type(s) as tic, bpc or xic (string or list of strings).')

        if isinstance(tmin, type(None)):
            tmin = df_l1.scan_start_time.min()
        if isinstance(tmax, type(None)):
            tmax = df_l1.scan_start_time.max()

        # if isinstance(ax, type(None)):
        #
        # else: ax1=ax
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        if xic:
            xx, xy = self.xic(xic_mz, xic_ppm, rt_min=tmin, rt_max=tmax)
            ax1.plot(xx, xy.astype(float), label='XIC ' + str(xic_mz) + 'm/z')
        if tic:
            ax1.plot(df_l1.scan_start_time, df_l1.total_ion_current.astype(float), label='TIC')
        if bpc:
            ax1.plot(df_l1.scan_start_time, df_l1.base_peak_intensity.astype(float), label='BPC')

        # ax1.set_title(df.fname[0], fontsize=8, y=-0.15)
        ax1.text(1.04, 0, df_l1.fname[0], rotation=90, fontsize=6, transform=ax1.transAxes)

        if df_l2.shape[0] == 0:
            print('Exp contains no CID data')
        else:
            ax1.scatter(df_l2.scan_start_time, np.zeros(df_l2.shape[0]), marker=3, c='black', s=6)
        fig.canvas.draw()
        offset = ax1.yaxis.get_major_formatter().get_offset()
        # offset = ax1.get_xaxis().get_offset_text()

        ax1.set_xlabel(r"$\bfs$")
        ax1.set_ylabel(r"$\bfCount$")
        ax2 = ax1.twiny()

        ax1.yaxis.offsetText.set_visible(False)
        # ax1.yaxis.set_label_text("original label" + " " + offset)
        if offset != '': ax1.yaxis.set_label_text(r"$\bfInt/count$ ($x 10^" + offset.replace('1e', '') + '$)')

        def tick_conv(X):
            V = X / 60
            return ["%.2f" % z for z in V]

        print(len(ax1.get_xticks()))
        # tick_loc = np.linspace(np.round(tmin/60), np.round(tmax/60),
        #                        (round(np.round((tmax - tmin) / 60, 0) / 2) + 1)) * 60
        tick_loc = np.arange(np.round(tmin/60), np.round(tmax/60), 2, dtype=float) * 60
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xticks(tick_loc)
        ax2.set_xticklabels(tick_conv(tick_loc))
        ax2.set_xlabel(r"$\bfmin$")
        # ax2.yaxis.set_label_text(r"$Int/count$ " + offset)

        ax1.legend(loc='best', bbox_to_anchor=(0.5, 0.5, 0.5, 0.5))

        fig.show()
        return fig, ax1, ax2