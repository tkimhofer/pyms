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

        if print_summary:
            print('---')
            print(f'Number of mzml files: {self.n_exp}')
            print(f'Files created from { self.df.mtime.iloc[mtime_min]} to { self.df.mtime.iloc[mtime_max]}')
            print(f'Files size ranging from {round(self.df.fsize_mb.iloc[fsize_mb_min],1)} MB to {round(self.df.fsize_mb.iloc[fsize_mb_max], 1)} MB')
            print(f'For detailed experiment list see obj.df')
            print('---')


class msExp:
    # import methods
    from ._peak_picking import est_intensities, cl_summary, feat_summary1, peak_picking, vis_feature_pp
    #from ._reading import read_mzml, read_bin, exp_meta, node_attr_recurse, rec_attrib, extract_mass_spectrum_lev1
    from ._isoPattern import element, element_list, calc_mzIsotopes

    def __init__(self, fpath, mslev='1', print_summary=False):
        self.fpath = fpath
        self.read_exp(mslevel=mslev, print_summary=print_summary)

    # def peak(self, st_len_min=10, plot=True, mz_bound=0.001, rt_bound=3, qc_par={'st_len_min': 5, 'raggedness': 0.15, 'non_neg': 0.8, 'sId_gap': 0.15, 'sino': 2, 'ppm': 15}):
    #     self.st_len_min=st_len_min
    #     self.qc_par = qc_par
    #     self.peak_picking(self.X2, self.st_len_min,)

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
        import pandas as pd
        # deconvolution based on isotopic mass differences (1 m/z)
        # check which features overlap in rt dimension

        rt_delta = 0.1

        descr = pd.DataFrame([self.feat[x]['descr'] for x in self.feat_l3])
        qc = pd.DataFrame([self.feat[x]['qc'] for x in self.feat_l3])
        quant = pd.DataFrame([self.feat[x]['quant'] for x in self.feat_l3])


        fs = pd.concat([descr, qc, quant], axis=1)
        fs.index = self.feat_l3

        fs['gr'] = np.ones(fs.shape[0]) * -1
        # test.sort_values(by=['sino'], ascending=False).id[0:30]

        S = self.feat_l3.copy()
        burn = []
        fgroups = {}
        c=0
        igrp = fs.shape[1]-1
        #for j in range(len(S)):
        while len(S) > 0:
            i = S[0]
            idx = np.where((fs['rt_maxI'].values > (fs.loc[i]['rt_maxI'] - rt_delta)) & (fs['rt_maxI'].values < (fs.loc[i]['rt_maxI'] + rt_delta)) & ~fs.index.isin(burn))[0]
            if len(idx) > 1:
                fs.iloc[idx, igrp]=c
                c += 1
                ids=fs.iloc[idx].index.values.tolist()
                burn = burn + ids
                [S.remove(x) for x in ids]
                ifound = {'seed': i, \
                          'ridx': ids,
                          'quant': fs.iloc[idx].raw.values,
                          'rt_imax': fs.iloc[idx].rt_maxI.values,
                          'mz_diff': fs.iloc[idx].mz_mean}
                fgroups.update({'fg'+ str(c) : ifound})
            else:
                S.remove(i)

        print(f'{c-1} feature groups detected')
        self.fgroups = fgroups
        self.fs = fs

    def vis_feature_gr(self, fgr, b_mz=4, b_rt=2):
        # l2 and l1 features mz/rt plot after peak picking
        # label feature with id
        # two panel plot: bottom m/z over st, top: st vs intentsity upon click on feature
        # basically like vis spectrum but including rectangles and all points but those in fgr feature(s) are s=0.1 and c='grey
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        from matplotlib.backend_bases import MouseButton
        from matplotlib.patches import Rectangle
        import numpy as np
        import pandas as pd
        from itertools import compress

        # which feature - define plot axis boundaries
        mz_min = self.fs.mz_min.loc[fgr['ridx']].min() - b_mz / 2
        mz_max = self.fs.mz_max.loc[fgr['ridx']].max() + b_mz / 2

        rt_min = self.fs.rt_min.loc[fgr['ridx']].min() - b_rt / 2
        rt_max = self.fs.rt_max.loc[fgr['ridx']].max() + b_rt / 2

        selection = {'mz_min': mz_min, 'mz_max': mz_max, 'rt_min': rt_min, 'rt_max': rt_max}

        def vis_feature(fdict, id, ax=None, add=False):
            import matplotlib.pyplot as plt

            # idx = np.where(self.Xf[:, 6] == fdict['id'])[0]
            if isinstance(ax, type(None)):
                fig = plt.figure()
                ax = fig.add_subplot(111)

            ax.plot(fdict['fdata']['st'], fdict['fdata']['I_raw'], label='raw', linewidth=0.5, color='black', zorder=0)

            if fdict['flev'] == 3:
                ax.plot(fdict['fdata']['st_sm'], fdict['fdata']['I_smooth'], label='smoothed', linewidth=0.5, color='cyan', zorder=0)
                ax.plot(fdict['fdata']['st_sm'], fdict['fdata']['I_bl'], label='baseline', linewidth=0.5, color='black', ls='dashed', zorder=0)
                ax.plot(fdict['fdata']['st_sm'], fdict['fdata']['I_sm_bline'], label='smoothed bl-cor', color='orange', zorder=0)
            if not add:
                ax.set_title(id + f',  flev: {fdict["flev"]}')
                ax.legend()
            else:
                old= axs[0].get_title()
                ax.set_title(old + '\n' +  id + f',  flev: {fdict["flev"]}')
            ax.set_ylabel(r"$\bfCount$")
            ax.set_xlabel(r"$\bfScan time$, s")

            data = np.array([list(fdict['qc'].keys()), np.round(list(fdict['qc'].values()), 2)]).T
            column_labels = ["descr", "value"]
            ax.table( cellText=data, colLabels=column_labels, loc='lower right', colWidths = [0.1]*3)

            return ax

        # Xf has 7 columns: 0 - scanId, 1 - mz, 2 - int, 3 - st, 4 - noiseBool, 5 - st_adj, 6 - clMem

        # which feature - define plot axis boundaries
        Xsub = self.window_mz_rt(self.Xf, selection, allow_none=False, return_idc=False)
        fid_window = np.unique(Xsub[:,6])
        l3_ffeat = list(compress(self.feat_l3, np.in1d([float(i.split(':')[1]) for i in self.feat_l3], fid_window)))

        cm = plt.cm.get_cmap('gist_ncar')
        fig, axs = plt.subplots(2,1, gridspec_kw={'height_ratios': [1, 2]}, figsize=(8,10), constrained_layout=False)
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        axs[0].set_ylabel(r"$\bfCount$")
        axs[0].set_xlabel(r"$\bfScan time$, s")

        vis_feature(self.feat[self.feat_l2[0]], id=self.feat_l2[0], ax=axs[0])
        axs[1].set_facecolor('#EBFFFF')
        axs[1].text(1.03, 0, self.df.fname[0], rotation=90, fontsize=6, transform=axs[1].transAxes)
        axs[1].scatter(Xsub[Xsub[:,4]==0, 3], Xsub[Xsub[:,4]==0, 1], s=0.1, c='gray', alpha=0.5)

        for i in l3_ffeat:
            fid = float(i.split(':')[1])
            # Xsub[Xsub[:,6] == fid, :]
            f1 = self.feat[i]
            a_col = 'grey'
            fsize = 6

            if any([i==x for x in fgr['ridx']]):
                a_col = 'green'
                fsize = 10

            mz=Xsub[Xsub[:, 6] == fid, 1]
            if len(mz) == len(f1['fdata']['st']):
                im = axs[1].scatter(f1['fdata']['st'], mz,
                                    c=np.log(f1['fdata']['I_raw']), s=5, cmap=cm)
                axs[1].add_patch(Rectangle(((f1['descr']['rt_min']), f1['descr']['mz_min']), \
                                           ((f1['descr']['rt_max']) - (f1['descr']['rt_min'])), \
                                           (f1['descr']['mz_max'] - f1['descr']['mz_min']), fc='#C2D0D6', \
                                           linestyle="solid", color='red', linewidth=2, zorder=0, in_layout=True,
                                           picker=False))
                axs[1].annotate(i.split(':')[1], (f1['descr']['rt_max'], f1['descr']['mz_max']),
                                bbox=dict(facecolor=a_col, alpha=0.3, boxstyle='circle', edgecolor='white',
                                          in_layout=False),
                                wrap=False, picker=True, fontsize=fsize)

        # coords = pd.DataFrame(coords)
        cbaxes = inset_axes(axs[1], width="30%", height="5%", loc=3)
        plt.colorbar(im, cax=cbaxes, ticks=[0., 1], orientation='horizontal')

        def p_text(event):
            print(event.artist)
            ids = str(event.artist.get_text())
            if event.mouseevent.button is MouseButton.LEFT:
                print('left click')
                print('id:' + str(ids))
                axs[0].clear()
                axs[0].set_title('')
                vis_feature(self.feat['id:' + ids], id=ids, ax=axs[0], add=False)
            if event.mouseevent.button is MouseButton.RIGHT:
                print('right click')
                print('id:' + str(ids))
                vis_feature(self.feat['id:' + ids], id=ids, ax=axs[0], add=True)

        cid1 = fig.canvas.mpl_connect('pick_event', p_text)
        axs[1].set_xlabel(r"$\bfScan time$, s")
        axs[1].yaxis.offsetText.set_visible(False)
        axs[1].yaxis.set_label_text(r"$\bfm/z$")

    def vis_fgroup(self, fgr, b_mz=2, b_rt=2, rt_bound=0.51, mz_bound=0.2):
        # visualise mz/rt plot after peak picking and isotope detection
        # basically like vis spectrum but including rectangles and all points but those in fgr feature(s) are s=0.1 and c='grey
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        import numpy as np

        # which feature - define plot axis boundaries
        mz_min = self.fs.mz_min.loc[fgr['ridx']].min() - b_mz / 2
        mz_max = self.fs.mz_max.loc[fgr['ridx']].max() + b_mz / 2

        rt_min = self.fs.rt_min.loc[fgr['ridx']].min() - b_rt / 2
        rt_max = self.fs.rt_max.loc[fgr['ridx']].max() + b_rt / 2

        Xsub = self.window_mz_rt(self.Xf,
                                 selection={'mz_min': mz_min, 'mz_max': mz_max, 'rt_min': rt_min, 'rt_max': rt_max},
                                 allow_none=False, return_idc=False)

        # define L2 clusters
        cl_l2 = self.fs.loc[fgr['ridx']]

        # define points exceeding noise and in L2 feature list
        # cl_n_above = np.where((Xsub[:, 4] == 0) & (np.isin(Xsub[..., 5], cl_l2.cl_id.values)))[0]
        # cl_n_below = np.where((Xsub[:, 4] == 1) | (~np.isin(Xsub[..., 5], cl_l2.cl_id.values)))[0]
        noi = np.where(Xsub[:, 4] == 1)[0]
        # assert ((len(cl_n_above) + len(cl_n_below)) == Xsub.shape[0])

        cm = plt.cm.get_cmap('gist_ncar')
        # fig, ax = plt.subplots()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.text(1.25, 0, self.df.fname[0], rotation=90, fontsize=6, transform=ax.transAxes)
        ax.scatter(Xsub[noi, 3], Xsub[noi, 1], s=0.5, c='gray', alpha=0.7)
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
    def tic_chromMS(self):

        import matplotlib.pyplot as plt
        import numpy as np
        df_l1 = self.df.loc[self.df.ms_level == '1']
        df_l2 = self.df.loc[self.df.ms_level == '2']
        if df_l1.shape[0] == 0: raise ValueError('mslevel 1 in df does not exist or is not defined')

        fig, axs = plt.subplots(2,1)
        axs[0].plot(df_l1.scan_start_time, df_l1.total_ion_current.astype(float), label='TIC')


        def mass_spectrum_callback(event):
            print(event.dblclick )
            if ((event.inaxes in [axs[0]]) & (event.dblclick == 1)):
                if event.xdata <= np.max(df_l1.scan_start_time):
                    idx_scan = np.argmin(np.abs(event.xdata - df_l1.scan_start_time))
                    axs[0].scatter(df_l1.scan_start_time[idx_scan], df_l1.total_ion_current.astype(float)[idx_scan], c='red')

                    ms_scan=float(self.df.id.iloc[idx_scan].split('scan=')[1])
                    idc = np.where(self.Xraw[:,0]==ms_scan)[0]


                    axs[1].clear()
                    ymax = np.max(self.Xraw[idc, 2])
                    y=self.Xraw[idc, 2]/ymax *100
                    x=self.Xraw[idc, 1]
                    axs[1].vlines(x, ymin=0, ymax=y)
                    axs[1].set_xlim(0, 1200)

                    # add labels of three max points
                    idc_max = np.argsort(-y)[0:3]
                    axs[1].set_ylim(0, 110)
                    for i in range(3):
                        axs[1].annotate(np.round(x[idc_max[i]], 3), (x[idc_max[i]], y[idc_max[i]]))

            else:
                return

        def zoom_massSp_callback(axx):
            print(axx.get_xlim())
            # if event.inaxes == axs[1]:
            #     print(axs[1].get_xlim())


        cid = fig.canvas.mpl_connect('button_press_event', mass_spectrum_callback)

        axs[1].callbacks.connect('xlim_changed', zoom_massSp_callback)


    def read_mzml(self, flag='1', print_summary=True):
        # read in data, files index 0 (see below)
        # flat is 1/2 for msLevel 1 or 2
        import pandas as pd
        import re
        import numpy as np
        import xml.etree.cElementTree as ET

        tree = ET.parse(self.fpath)
        root = tree.getroot()

        child = children(root)
        imzml = child.index('mzML')
        mzml_children = children(root[imzml])

        obos = root[imzml][mzml_children.index('cvList')]
        self.obo_ids = get_obo(obos, obo_ids={})

        seq = np.where(~np.isin(mzml_children, ['cvList', 'run']))[0]

        pp = {}
        for j in seq:
            filed = node_attr_recurse(s=root[imzml][j], d=4, c=0, ii=[])
            dn = {}
            for i in range(len(filed)):
                dn.update(dict(zip([filed[i]['path'] + '_' + re.sub('\{.*\}', '', x) for x in list(filed[i].keys())[1:]],
                                   list(filed[i].values())[1:])))
            pp.update({mzml_children[j]: dn})

        run = root[imzml][mzml_children.index('run')]
        # out=collect_spectra_chrom(s, ii={}, d=9, c=0, flag='1', tag='')
        out31 = collect_spectra_chrom(s=run, ii={}, d=20, c=0, flag=flag, tag='', obos=self.obo_ids)

        # summarise and collect data
        # check how many msl1 and ms2

        ms = []
        time = []
        cg = []
        meta = []
        for i in out31:
            meta.append(out31[i]['meta'])
            if ('s' in i) and (out31[i]['meta']['MS:1000511'] == '1'):
                dd = []
                for j in (out31[i]['data']):
                    dd.append(out31[i]['data'][j]['d'])
                ms.append(dd)
                # time.append(np.ones_like(out31[i]['data']['m/z']['d']) * float(out31[i]['meta']['MS:1000016']))
                time.append(float(out31[i]['meta']['MS:1000016']))
            if 'c' in i:
                dd = []
                for j in (out31[i]['data']):
                    dd.append(out31[i]['data'][j]['d'])
                cg.append(dd)

        df = pd.DataFrame(meta, index=list(out31.keys()))
        df.columns = [self.obo_ids[x]['name'].replace(' ', '_') if x in self.obo_ids.keys() else x for x in df.columns.values]
        df['fname'] = self.fpath
        # summarise columns
        summary = df.describe().T

        self.dstype='Ukwn'
        # if all chromatograms -> targeted assay
        if all(['c' in x for x in out31.keys()]):
            self.dstype = 'srm, nb of functions: ' + str(len(cg))

            # collect chromatogram data
            idx=np.where((df['selected_reaction_monitoring_chromatogram']==True).values)[0]
            le = len(cg[idx[0]])
            if le != 2:
                raise ValueError('Check SRM dataArray length')
            stime = [cg[0] for i in idx]
            Int = [cg[1] for i in idx]

            return (self.dstype, df, stime, Int, summary)

        else:
            dstype = 'untargeted assay'
            if list(out31['s1']['data'].keys())[0] == 'm/z':
                mz = np.concatenate([x[0] for x in ms])
            if list(out31['s1']['data'].keys())[1] == 'Int':
                Int = np.concatenate([x[1] for x in ms])

            size = df[df.ms_level == '1'].defaultArrayLength.astype(int).values
            idx_time = np.argsort(time)
            scanid = np.concatenate([np.ones(size[i]) * idx_time[i] for i in range(len(size))])
            stime = np.concatenate([np.ones(size[i]) * time[i] for i in range(len(size))])

            if len(np.unique([len(scanid), len(mz), len(Int), len(stime)])) > 1:
                raise ValueError('Mz, rt or scanid dimensions not matching - possibly corrupt data,')

            if print_summary:
                print(dstype)
                print(summary)

            return (dstype, df, scanid, mz, Int, stime, summary)

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
                 selection={'mz_min': None, 'mz_max': None, 'rt_min': None, 'rt_max': None},
                   qc_par={'st_len_min': 5, 'raggedness': 0.15, 'non_neg': 0.8, 'sId_gap': 0.15, 'sino': 2, 'ppm': 15},
                   multicore=True):

        self.qc_par = qc_par
        import numpy as np
        print('Start peak picking...')

        def pp(ex, st_adj, q_noise, dbs_par, selection):
            try:
                ex.peak_picking(st_adj, q_noise, dbs_par, selection)
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
            if hasattr(ex, 'feat_l3'):
                print(e)
                fl2 = ex.feat_l3
                rt=[]
                mz=[]
                for i in fl2:
                    rt.append(ex.feat[i]['descr']['rt_maxI'])
                    mz.append(ex.feat[i]['descr']['mz_maxI'])
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


def rec_attrib(s, ii={}):
    # recursively collecting cvParam
    for at in list(s):
        if 'cvParam' in at.tag:
            if 'name' in at.attrib.keys():
                ii.update({at.attrib['name'].replace(' ', '_'): at.attrib['value']})
        if len(list(at)) > 0:
            rec_attrib(at, ii)

    return ii

def extract_mass_spectrum_lev1(i, s_level, s_dic):
    import base64
    import numpy as np
    import zlib

    s = s_level[i]
    iistart = s_dic[i]
    ii = rec_attrib(s, iistart)
    # Extract data
    # two binary arrays: m/z value and intensity
    # read m/z value
    le = len(list(s)) - 1
    # pars = list(list(list(s)[7])[0]) # ANPC files
    pars = list(list(list(s)[le])[0])  # NPC files

    dt = arraydtype(pars)

    # check if compression
    if pars[1].attrib['name'] == 'zlib compression':
        mz = np.frombuffer(zlib.decompress(base64.b64decode(pars[3].text)), dtype=dt)
    else:
        mz = np.frombuffer(base64.b64decode(pars[3].text), dtype=dt)

    # read intensity values
    pars = list(list(list(s)[le])[1])
    dt = arraydtype(pars)
    if pars[1].attrib['name'] == 'zlib compression':
        I = np.frombuffer(zlib.decompress(base64.b64decode(pars[3].text)), dtype=dt)
    else:
        I = np.frombuffer(base64.b64decode(pars[3].text), dtype=dt)

    return (ii, mz, I)


def collect_spectra_chrom(s, ii={}, d=9, c=0, flag='1', tag='', obos={}):
    import re
    if c == d: return
    if (('chromatogram' == tag) | ('spectrum' == tag)):
        rtype = 'Ukwn'
        if ('chromatogram' == tag):
            rtype = 'c'
        if ('spectrum' == tag):
            rtype = 's'
        add_meta, data = extr_spectrum_chromg(s, flag=flag, rtype=rtype, obos=obos)
        # if rtype == 'c': print(tag); print(add_meta); print(data)
        ii.update({rtype + add_meta['index']: {'meta': add_meta, 'data': data}})

    ss = list(s)
    if len(ss) > 0:
        for i in range(len(ss)):
            tag = re.sub('\{.*\}', '', ss[i].tag)
            collect_spectra_chrom(ss[i], ii=ii, d=d, c=c + 1, flag=flag, tag=tag, obos=obos)

    return ii

def extr_spectrum_chromg(x, flag='1', rtype='s', obos=[]):
    add_meta = vaPar_recurse(x, ii={}, d=6, c=0, dname='name', dval='value', ddt='all')

    if rtype == 's':  # spectrum
        if flag == '1':  # mslevel 1 input
            if ('MS:1000511' in add_meta.keys()):  # find mslevel information
                if add_meta['MS:1000511'] == '1':  # read in ms1
                    out = data_recurse1(x, ii={}, d=9, c=0, ip='', obos=obos)
                else:
                    return (add_meta, [])
            else:  # can;t find mslevel info
                print('MS level information not found not found - reading all experiments.')
                out = data_recurse1(x, ii={}, d=9, c=0, ip='', obos=obos)
        else:
            out = data_recurse1(x, ii={}, d=9, c=0, ip='', obos=obos)
    else:
        out = data_recurse1(x, ii={}, d=9, c=0, ip='', obos=obos)
    return (add_meta, out)

def data_recurse1(s, ii={}, d=9, c=0, ip='', obos=[]):
    import re
    # recursively extracts attributes from node s and with depth  d, add label children iterator as prefix
    if c == d: return
    if ('binaryDataArray' == re.sub('\{.*\}', '', s.tag)):
        iis, ft = read_bin(s, obos)
        ii.update({ip + ft: iis})
    # if ('cvParam' in s.tag):
    ss = list(s)
    if len(ss) > 0:
        for i in range(len(ss)):
            tag = re.sub('\{.*\}', '', ss[i].tag)
            if ('chromatogram' == tag):
                ip = 'c'
            if ('spectrum' == tag):
                ip = 's'
            data_recurse1(s=ss[i], ii=ii, d=d, c=c + 1, ip=ip, obos=obos)
    return ii

def dt_co(dvars):
    import numpy as np
    # dtype and compression
    dt = None  # data type
    co = None  # compression
    ft = 'ukwn'  # feature type
    if 'MS:1000523' in dvars.keys():
        dt = np.dtype('<d')

    if 'MS:1000522' in dvars.keys():
        dt = np.dtype('<i8')

    if 'MS:1000521' in dvars.keys():
        dt = np.dtype('<f')

    if 'MS:1000519' in dvars.keys():
        dt = np.dtype('<i4')

    if isinstance(dt, type(None)):
        raise ValueError('Unknown variable type')

    if 'MS:1000574' in dvars.keys():
        co = 'zlib'

    if 'MS:1000514' in dvars.keys():
        ft = 'm/z'
    if 'MS:1000515' in dvars.keys():
        ft = 'Int'

    if 'MS:1000595' in dvars.keys():
        ft = 'time'
    if 'MS:1000516' in dvars.keys():
        ft = 'Charge'
    if 'MS:1000517' in dvars.keys():
        ft = 'sino'

    return (dt, co, ft)

def read_bin(k, obo_ids):
    # k is binary array
    import numpy as np
    import zlib
    import base64
    # collect metadata
    dvars = vaPar_recurse(k, ii={}, d=3, c=0, dname='accession', dval='cvRef', ddt='-')
    dt, co, ft = dt_co(dvars)

    child = children(k)
    dbin = k[child.index('binary')]

    if co == 'zlib':
        d = np.frombuffer(zlib.decompress(base64.b64decode(dbin.text)), dtype=dt)
    else:
        d = np.frombuffer(base64.b64decode(dbin.text), dtype=dt)
    # return data and meta
    out = {'i': [{x: obo_ids[x]['name']} for x in dvars.keys() if x in obo_ids.keys()], 'd': d}

    return (out, ft)

def vaPar_recurse(s, ii={}, d=9, c=0, dname='accession', dval='value', ddt='all'):
    import re
    # recursively extracts attributes from node s and with depth  d, add label children iterator as prefix
    if c == d: return
    if ('cvParam' in s.tag):
        iis = {}
        name = s.attrib[dname]
        value = s.attrib[dval]
        if value == '': value = True
        iis.update({name: value})
        if 'unitCvRef' in s.attrib:
            iis.update({s.attrib['unitAccession']: s.attrib['unitName']})
        ii.update(iis)
    else:
        if ddt == 'all':
            iis = s.attrib
            ii.update(iis)
    # if ('cvParam' in s.tag):
    ss = list(s)
    if len(ss) > 0:
        # if ('spectrum' in s.tag):
        for i in range(len(ss)):
            vaPar_recurse(s=ss[i], ii=ii, d=d, c=c + 1, ddt=ddt)
    return ii

def get_obo(obos, obo_ids={}):
    # download obo annotation data
    # create single dict with keys being of obo ids, eg, MS:1000500
    # obos is first cv element in mzml node
    import obonet
    for o in obos:
        id=o.attrib['id']
        if id not in obo_ids.keys():
            gr = obonet.read_obo(o.attrib['URI'])
            gv = gr.nodes(data=True)
            map = {id_: {'name': data.get('name'), 'def': data.get('def'), 'is_a': data.get('is_A')} for
                          id_, data in gv}
            #obo_ids.update({id: {'inf': o.attrib, 'data': map}})
            obo_ids.update(map)

    return obo_ids

def children(xr):
    import re
    return [re.sub('\{.*\}', '', x.tag) for x in list(xr)]

def exp_meta(root, ind, d):
    # combine attributes to dataframe with column names prefixed acc to child/parent relationship
    # remove web links in tags
    import pandas as pd
    import re
    filed = node_attr_recurse(s=root, d=d, c=0, ii=[])

    dn = {}
    for i in range(len(filed)):
        dn.update(dict(zip([filed[i]['path'] + '_' + re.sub('\{.*\}', '', x) for x in list(filed[i].keys())[1:]],
                           list(filed[i].values())[1:])))
    return pd.DataFrame(dn, index=[ind])

def node_attr_recurse(s, ii=[], d=3, c=0,  pre=0):
    import re
    # recursively extracts attributes from node s and with depth  d, add label children iterator as prefix
    if c == d: return
    # define ms level
    if ('list' not in s.tag) | ('spectrum' not in s.tag):
        iis = {}
        iis['path'] = re.sub('\{.*\}', '', s.tag) + '_' + str(pre)
        iis.update(s.attrib)
        ii.append(iis)
    if len(list(s)) > 0:
        if ('spectrum' not in s.tag.lower()):
            ss = list(s)
            for i in range(len(list(ss))):
                at = ss[i]
                node_attr_recurse(s=at, ii=ii,d=d, c=c + 1, pre=i)
    return ii

def read_exp(path, ftype='mzML', plot_chrom=False, n_max=1000, only_summary=False, d=4):
    import subprocess
    import xml.etree.ElementTree as ET

    cmd = 'find ' + path + ' -type f -iname *'+ ftype
    sp = subprocess.getoutput(cmd)
    files = sp.split('\n')

    if len(files) == 0:
        raise ValueError(r'No .' + ftype + ' files found in path: '+path)

    if len(files) > n_max:
        files=files[0:n_max]

    if only_summary:
        print('Creating experiment summary.')
        #import hashlib
        #hashlib.md5(open(fid,'rb').read()).hexdigest()

        df_summary = []
        for i in range(len(files)):
            print(files[i])
            tree = ET.parse(files[i])
            root = tree.getroot()
            #tt = exp_meta(root=root, ind=i, d=d).T
            df_summary.append(exp_meta(root=root, ind=i, d=d))

        df = pd.concat(df_summary).T
        return df
    else:

        data=[]
        df_summary = []
        print('Importing '+ str(len(files)) + ' files.')
        for i in range(len(files)):
            tree = ET.parse(files[i])
            root = tree.getroot()
            df_summary.append(exp_meta(root=root, ind=i, d=d))
            out = read_mzml_file(root, ms_lev=1, plot=plot_chrom)
            data.append(out[0:3])

        df = pd.concat(df_summary).T
        return (df, data)

def read_mzml_file(root, ms_lev=1, plot=True):
    from tqdm import tqdm
    import time
    import numpy as np
    import pandas as pd
    # file specific variables
    le = len(list(root[0]))-1
    id = root[0][le].attrib['id']


    StartTime=root[0][le].attrib['startTimeStamp'] # file specific
    #ii=root[0][6][0].attrib

    # filter ms level spectra
    s_all = list(root[0][le][0])

    s_level=[]
    s_dic=[]
    for s in s_all:
        scan = s.attrib['id'].replace('scan=', '')
        arrlen = s.attrib['defaultArrayLength']

        ii = {'fname': id, 'acquTime': StartTime, 'scan_id': scan, 'defaultArrayLength': arrlen}
        if 'name' in s[0].attrib.keys():
            if ((s[0].attrib['name'] == 'ms level') & (s[0].attrib['value'] == str(ms_lev))) | (s[0].attrib['name'] == 'MS1 spectrum'):
                # desired ms level
                s_level.append(s)
                s_dic.append(ii)
    print('Number of MS level '+ str(ms_lev)+ ' mass spectra : ' + str(len(s_level)))
    # loop over spectra
    # # check multicore
    # import multiprocessing
    # from joblib import Parallel, delayed
    # from tqdm import tqdm
    # import time
    # ncore = multiprocessing.cpu_count() - 2
    # #spec_seq = tqdm(np.arange(len(s_level)))
    # spec_seq = tqdm(np.arange(len(s_level)))
    # a= time.time()
    # bls = Parallel(n_jobs=ncore, require='sharedmem')(delayed(extract_mass_spectrum_lev1)(i, s_level, s_dic) for i in spec_seq)
    # b=time.time()
    # extract mz, I ,df
    # s1=(b-a)/60

    spec_seq = np.arange(len(s_level))
    #spec_seq = np.arange(10)
    ss = []
    I = []
    mz = []
    a = time.time()
    for i in tqdm(spec_seq):
        ds, ms, Is = extract_mass_spectrum_lev1(i, s_level, s_dic)
        ss.append(ds)
        mz.append(ms)
        I.append(Is)

    b = time.time()
    s2 = (b - a) / 60

    df = pd.DataFrame(ss)

    swll = df['scan_window_lower_limit'].unique()
    swul = df['scan_window_upper_limit'].unique()

    if len(swll) ==1 & len(swul) == 1:
        print('M/z scan window: ', swll[0] + "-" + swul[0] + ' m/z')
    else:
        print('Incosistent m/z scan window!')
        print('Lower bound: '+str(swll)+' m/z')
        print('Upper bound: '+str(swul)+' m/z')


    #df.groupby('scan_id')['scan_start_time'].min()

    df.scan_start_time=df.scan_start_time.astype(float)

    tmin=df.scan_start_time.min()
    tmax=df.scan_start_time.max()
    print('Recording time: ' + str(np.round(tmin,1)) + '-' + str(np.round(tmax,1)), 's (' + str(np.round((tmax-tmin)/60, 1)), 'm)')
    print('Read-in time: ' + str(np.round(s2, 1)) + ' min')
    # df.scan_id = df.scan_id.astype(int)
    df.scan_id = np.arange(1, df.shape[0]+1)
    df.total_ion_current=df.total_ion_current.astype(float)
    df.base_peak_intensity=df.base_peak_intensity.astype(float)

    # if plot:
    #     pl = plt_chromatograms(df, tmin, tmax)
    #     return (mz, I, df, pl)
    # else:
    return (mz, I, df)

def arraydtype(pars):
    import numpy as np
    # determines base64 single or double precision, all in little endian
    pdic = {}
    for pp in pars:
        if 'name' in pp.attrib.keys():
            pdic.update({pp.attrib['accession'].replace(' ', '_'): pp.attrib['name']})
        else:
            pdic.update(pp.attrib)

    dt = None
    if 'MS:1000523' in pdic.keys():
        dt = np.dtype('<d')

    if 'MS:1000522' in pdic.keys():
        dt = np.dtype('<i8')

    if 'MS:1000521' in pdic.keys():
        dt = np.dtype('<f')

    if 'MS:1000519' in pdic.keys():
        dt = np.dtype('<i4')

    if isinstance(dt, type(None)):
        raise ValueError('Unknown variable type')

    return dt
