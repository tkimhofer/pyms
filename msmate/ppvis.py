
def vis_feature_ppf(self, selection: dict = {'mz_min': 425, 'mz_max': 440, 'rt_min': 400, 'rt_max': 440}, lev: int = 3, q_Int=0.5):
    # l2 and l1 features mz/rt plot after peak picking
    # label feature with id
    # two panel plot: bottom m/z over st, top: st vs intentsity upon click on feature
    # basically like vis spectrum but including rectangles and all points but those in fgr feature(s) are s=0.1 and c='grey

    @log
    def vis_feature(fdict, id, ax=None, add=False):
        # idx = np.where(self.Xf[:, 6] == fdict['id'])[0]
        if isinstance(ax, type(None)):
            fig = plt.figure()
            ax = fig.add_subplot(111)

        ax.plot(fdict['fdata']['st'], fdict['fdata']['I_raw'], label='raw', linewidth=0.5, color='black', zorder=0)

        if fdict['flev'] == 3:
            ax.plot(fdict['fdata']['st_sm'], fdict['fdata']['I_smooth'], label='smoothed', linewidth=0.5,
                    color='cyan', zorder=0)
            ax.plot(fdict['fdata']['st_sm'], fdict['fdata']['I_bl'], label='baseline', linewidth=0.5,
                    color='black', ls='dashed', zorder=0)
            ax.plot(fdict['fdata']['st_sm'], fdict['fdata']['I_sm_bline'], label='smoothed bl-cor',
                    color='orange', zorder=0)
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

    # Xf has 7 columns: 0 - scanId, 1 - mz, 2 - int, 3 - st, 4 - noiseBool, 5 - st_adj, 6 - clMem

    # which feature - define plot axis boundaries
    Xsub = self.window_mz_rt(self.Xf, selection, allow_none=False, return_idc=False)
    fid_window = np.unique(Xsub[:, 6])
    l2_ffeat = list(compress(self.feat_l2, np.in1d([float(i.split(':')[1]) for i in self.feat_l2], fid_window)))
    l3_ffeat = list(compress(self.feat_l3, np.in1d([float(i.split(':')[1]) for i in self.feat_l3], fid_window)))

    cm = plt.cm.get_cmap('gist_ncar')
    fig, axs = plt.subplots(2, 2, gridspec_kw={'height_ratios': [1, 2]}, figsize=(8, 10), constrained_layout=False, sharey=True)
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    axs[0,0].set_ylabel(r"$\bfCount$")
    axs[0,0].set_xlabel(r"$\bfScan time$, s")

    vis_feature(self.feat[self.feat_l2[0]], id=self.feat_l2[0], ax=axs[0])
    axs[1,0].set_facecolor('#EBFFFF')
    axs[1,0].text(1.03, 0, self.fname, rotation=90, fontsize=6, transform=axs[1].transAxes)
    axs[1,0].scatter(Xsub[Xsub[:, 4] == 0, 3], Xsub[Xsub[:, 4] == 0, 1], s=0.1, c='gray', alpha=0.5)

    # fill axis for raw data results
    idc_above = np.where(Xsub[:, 2] > noise_thres)[0]
    idc_below = np.where(Xsub[:, 2] <= noise_thres)[0]

    cm = plt.cm.get_cmap('rainbow')
    axs[1,1].scatter(Xsub[idc_below, 3], Xsub[idc_below, 1], s=0.1, c='gray', alpha=0.5)
    ims = ax.scatter(Xsub[idc_above, 3], Xsub[idc_above, 1], c=np.log(Xsub[idc_above, 2]), s=5, cmap=cm)
    # fig.canvas.draw()
    fig.colorbar(im, ax=axs[1,1])

    axs[1,1].set_xlabel(r"$\bfScan time$ (sec)")


    if lev < 3:
        for i in l2_ffeat:
            fid = float(i.split(':')[1])
            f1 = self.feat[i]
            # if (f1['flev'] == 2):
            a_col = 'red'
            fsize = 6
            mz = Xsub[Xsub[:, 6] == fid, 1]
            if len(mz) == len(f1['fdata']['st']):
                im = axs[1,0].scatter(f1['fdata']['st'], mz,
                                    c=np.log(f1['fdata']['I_raw']), s=5, cmap=cm)
                axs[1,0].add_patch(Rectangle(((f1['descr']['rt_min']), f1['descr']['mz_min']), \
                                           ((f1['descr']['rt_max']) - (f1['descr']['rt_min'])), \
                                           (f1['descr']['mz_max'] - f1['descr']['mz_min']), fc='#C2D0D6', \
                                           linestyle="solid", color='red', linewidth=2, zorder=0,
                                           in_layout=True,
                                           picker=False))
                axs[1, 0].annotate(i.split(':')[1], (f1['descr']['rt_max'], f1['descr']['mz_max']),
                                bbox=dict(facecolor=a_col, alpha=0.3, boxstyle='circle', edgecolor='white',
                                          in_layout=False),
                                wrap=False, picker=True, fontsize=fsize)

    if lev == 3:
        for i in l3_ffeat:
            print(i)
            fid = float(i.split(':')[1])
            f1 = self.feat[i]
            if (f1['flev'] == 3):
                print('yh')
                a_col = 'green'
                fsize = 10
                mz = Xsub[Xsub[:, 6] == fid, 1]
                if len(mz) == len(f1['fdata']['st']):
                    axs[1, 0].add_patch(Rectangle(((f1['descr']['rt_min']), f1['descr']['mz_min']), \
                                               ((f1['descr']['rt_max']) - (f1['descr']['rt_min'])), \
                                               (f1['descr']['mz_max'] - f1['descr']['mz_min']), fc='#C2D0D6', \
                                               linestyle="solid", color='red', linewidth=2, zorder=0,
                                               in_layout=True,
                                               picker=False))
                    axs[1, 0].annotate(i.split(':')[1], (f1['descr']['rt_max'], f1['descr']['mz_max']),
                                    bbox=dict(facecolor=a_col, alpha=0.3, boxstyle='circle', edgecolor='white',
                                              in_layout=False),
                                    wrap=False, picker=True, fontsize=fsize)
                    im = axs[1, 0].scatter(f1['fdata']['st'], mz,
                                        c=np.log(f1['fdata']['I_raw']), s=5, cmap=cm)
    cbaxes = inset_axes(axs[1, 0], width="30%", height="5%", loc=3)
    plt.colorbar(im, cax=cbaxes, ticks=[0., 1], orientation='horizontal')

    def p_text(event):
        # print(event.artist)
        ids = str(event.artist.get_text())
        if event.mouseevent.button is MouseButton.LEFT:
            # print('left click')
            # print('id:' + str(ids))
            axs[0, 0].clear()
            axs[0, 0].set_title('')
            vis_feature(self.feat['id:' + ids], id=ids, ax=axs[0], add=False)
            event.canvas.draw()
        if event.mouseevent.button is MouseButton.RIGHT:
            # print('right click')
            # print('id:' + str(ids))
            vis_feature(self.feat['id:' + ids], id=ids, ax=axs[0], add=True)
            event.canvas.draw()

    cid1 = fig.canvas.mpl_connect('pick_event', p_text)
    axs[1, 0].set_xlabel(r"$\bfScan time$, s")
    axs[1, 0].yaxis.offsetText.set_visible(False)
    axs[1, 0].yaxis.set_label_text(r"$\bfm/z$")


class Exp1:
    def __init__(self, fname, type, mslevel, smode, amode, seg, verbosity=0):
        import sqlite3, os
        import baf2sql
        import pickle
        self.fname = fname
        self.type = type
        self.mslevel = str(mslevel) if not isinstance(mslevel, list) else tuple(map(str, mslevel))
        self.smode = str(smode) if not isinstance(smode, list) else tuple(map(str,smode))
        if len(self.smode) ==1:
            self.smode = self.smode[0]
        self.amode = str(amode)
        self.seg = str(seg) if not isinstance(seg, list) else tuple(map(str, seg))
        if len(self.seg) ==1:
            self.seg = self.seg[0]

        self.infile = os.path.join('/data', self.fname, "analysis.baf")
        # if verbosity == 1:
        #     print('connecting to baf binary storage')
        self.bs = baf2sql.BinaryStorage(self.infile)
        # if verbosity == 1:
        #     print('sqlitecache filename')
        sqlite_fn = baf2sql.getSQLiteCacheFilename(self.infile)
        # if verbosity == 1:
        #     print('establish sqlite connection')
        self.con = sqlite3.connect(sqlite_fn)
        self.con.row_factory = sqlite3.Row
        self.outfile = os.path.join('/data', self.fname, f'mm8v3_{self.type}_msl{self.mslevel}_sm{"".join(self.smode)}_am{self.amode}_seg{"".join(self.seg)}.p')
        print(self.outfile)

        self.properties = None
        self.nSpectra = None
        self.sdata = None

        self.collectProp()

        if type == 'meta':
            self.collectMeta()
        else:
            self.collectSpec()
        pickle.dump(self.edata, open(self.outfile, 'wb'))
        self.con.close()


    def collectProp(self):
        q=self.con.execute(f"select * from Properties")
        res = [dict(x) for x in q.fetchall()]
        prop = {}; [prop.update({x['Key']: x['Value']}) for x in res]
        self.properties = prop


    def collectMeta(self):
        q = self.con.execute(f"select * from SupportedVariables")
        vlist = [dict(x) for x in q.fetchall()]
        vdict = {x['Variable']: x for x in vlist}
        qst = f"SELECT * FROM Spectra as s JOIN AcquisitionKeys as ak ON s.AcquisitionKey=ak.Id WHERE Segment " \
              f"{'='if not isinstance(self.seg, tuple) else 'in'} {self.seg} AND " \
              f"ScanMode {'='if not isinstance(self.smode, tuple) else 'in'} {self.smode} AND" \
              f"MsLevel {'='if not isinstance(self.mslevel, tuple) else 'in'} {self.mslevel} AND ORDER BY Rt"
        res = [dict(x) for x in self.con.execute(qst).fetchall()]
        self.nSpectra = len(res)
        cvar = [x for x in list(res[0].keys()) if ('Id' in x)]
        edat = {x: [] for x in cvar}
        for i in range(self.nSpectra):
            mets = {x: res[i][x] for x in res[i] if not ('Id' in x)}
            q = self.con.execute(f"select * from PerSpectrumVariables WHERE Spectrum = {i}")
            specVar = [dict(x) for x in q.fetchall()]
            mets.update({vdict[k['Variable']]['PermanentName']: k['Value'] for k in specVar})
            edat[cvar[0]].append(mets)
        self.edata = edat

    def collectSpec(self):
        q = self.con.execute(f"select * from SupportedVariables")
        vlist = [dict(x) for x in q.fetchall()]
        vdict = {x['Variable']: x for x in vlist}
        qst = f"SELECT * FROM Spectra as s JOIN AcquisitionKeys as ak ON s.AcquisitionKey=ak.Id WHERE Segment " \
              f"{'=' if not isinstance(self.seg, tuple) else 'in'} {self.seg} AND " \
              f"ScanMode {'=' if not isinstance(self.smode, tuple) else 'in'} {self.smode} AND" \
              f"MsLevel {'='if not isinstance(self.mslevel, tuple) else 'in'} {self.mslevel} AND ORDER BY Rt"
        print(qst)
        res = [dict(x) for x in self.con.execute(qst).fetchall()]
        self.nSpectra = len(res)
        cvar = ['sinfo', 'LineIndexId', 'LineMzId', 'LineIntensityId']
        edat = {x: [] for x in cvar}
        for i in range(self.nSpectra):
            mets = {x: res[i][x] for x in res[i] if not ('Id' in x)}
            q = self.con.execute(f"select * from PerSpectrumVariables WHERE Spectrum = {res[i]['Id']}")
            specVar = [dict(x) for x in q.fetchall()]
            mets.update({vdict[k['Variable']]['PermanentName']: k['Value'] for k in specVar})
            mets.update({'Id': res[i]['Id']})
            edat[cvar[0]].append(mets)
            for j in cvar[1:]:
                if (res[i][j] is not None):
                    edat[j].append((res[i]['Id'], self.bs.readArrayDouble(res[i][j])))
        self.edata = edat