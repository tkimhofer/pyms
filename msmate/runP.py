from typeguard import typechecked
import logging
import functools
from typing import Union

import os
import numpy as np
import pandas as pd
import pickle
import time
import docker as dock

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.backend_bases import MouseButton
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterSciNotation

from itertools import compress

from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks, peak_prominences
from scipy import integrate


#
# import argparse
#
# argparse.ArgumentParser

# lev = logging.DEBUG
lev = logging.INFO
logging.basicConfig(level=lev)
logger = logging.getLogger()

def log(f):
    @functools.wraps(f)
    def wr(*arg, **kwargs):
        try:
            out = f(*arg, **kwargs)
            return out
        except Exception as e:
            logger.exception(f'{f.__name__}: {repr(e)}')
    return wr


class ExpSum:
    def __init__(self, fname: str):
        import sqlite3, os
        self.fname = fname
        self.sqlite_fn= os.path.join(self.fname, "analysis.sqlite")
        print(self.sqlite_fn)
        if not os.path.isfile(self.sqlite_fn):
            raise SystemError('Sqlite file not found')
        self.c = sqlite3.connect(self.sqlite_fn)
        self.c.row_factory = sqlite3.Row

        self.properties = None
        self.queryProp()
        self.collectMeta()
        self.c.close()
    def uniqueLevs(self):
        df2q = pd.DataFrame(
            [(str(x['MsLevel']) + str(x['AcquisitionKey']) +str(x['AcquisitionMode']) + str(x['ScanMode']) + '_' + str(x['Collision_Energy_Act'])) for x in self.edat['Id'] if x['Segment'] == 3],
            columns=['ML_'+'AK_'+'AM_'+'SM_'+'CE_eV'])
        print(df2q.value_counts())
        return df2q
    def queryProp(self):
        q=self.c.execute(f"select * from Properties")
        res = [dict(x) for x in q.fetchall()]
        prop = {}; [prop.update({x['Key']: x['Value']}) for x in res]
        self.properties = prop
    def collectMeta(self):
        import pandas as pd
        import numpy as np
        q = self.c.execute(f"select * from SupportedVariables")
        vlist = [dict(x) for x in q.fetchall()]
        vdict = {x['Variable']: x for x in vlist}
        qst = "SELECT * FROM Spectra as s JOIN AcquisitionKeys as ak ON s.AcquisitionKey=ak.Id ORDER BY Rt"
        res = [dict(x) for x in self.c.execute(qst).fetchall()]
        tt=pd.DataFrame(res)
        self.nSpectra = len(res)
        cvar = [x for x in list(res[0].keys()) if ('Id' in x)]
        edat = {x: [] for x in cvar}
        for i in range(self.nSpectra):
            mets = {x: res[i][x] for x in res[i] if not ('Id' in x)}
            q = self.c.execute(f"select * from PerSpectrumVariables WHERE Spectrum = {i}")
            specVar = [dict(x) for x in q.fetchall()]
            mets.update({vdict[k['Variable']]['PermanentName']: k['Value'] for k in specVar})
            edat[cvar[0]].append(mets)
        self.edat = edat
        self.ds=pd.DataFrame(self.edat['Id'])
        self.scantime =  self.ds.Rt
        polmap = {0: 'P', 1: 'N'}
        df1 = pd.DataFrame(
            [(polmap[x['Polarity']], str(x['MzAcqRangeLower']) + '-' + str(x['MzAcqRangeUpper']), x['MsLevel']) for x in
             self.edat['Id']], columns=['Polarity', 'MzAcqRange', 'MsLevel'])
        df1 = pd.DataFrame([self.csummary(i, df1) for i in range(df1.shape[1])], index=df1.columns).transpose()
        df1['nSc'] = len(self.edat['Id'])

        df2 = pd.DataFrame([(x['MsLevel'], x['AcquisitionKey'], x['AcquisitionMode'], x['ScanMode'], x['Segment']) for x in self.edat['Id']], columns=['MsLevel', 'AcqKey', 'AcqMode', 'ScMode', 'Segments'])
        t = pd.DataFrame(df2.groupby(['Segments']).apply(lambda x: self.vc(x, n=df2.shape[0])))
        t = t.rename(index={t.index[i]: 'Segment ' + str(t.index[i]) for i in range(t.shape[0])}).transpose()
        asps = 1 / ((self.scantime.max() - self.scantime.min()) / df1['nSc'].iloc[0])  # hz
        ii = np.diff(np.unique(self.scantime))
        sps = pd.DataFrame(
            {'ScansPerSecAver': asps, 'ScansPerSecMin': 1 / ii.max(), 'ScansPerSecMax': 1 / ii.min()}, index=[0])
        self.summary = pd.concat([df1, t, sps], axis=1).T
        print(self.summary)
        df2q=self.uniqueLevs()
        if df2q.shape[0] == 0:
            print('No Segment 3 data')
        else:
            print(df2q.shape)
            print(self.ds.shape)
            ds3 = self.ds[self.ds.Segment == 3].copy()
            ds3['ulev'] = df2q.iloc[:,0].values
            self.plt_chromatogram(ds3)
    def plt_chromatogram(self, ds3):
        fig = plt.figure()
        lun = ds3['ulev'].value_counts()
        lun=lun[lun > 10].index.values
        axs = fig.subplots(len(lun), 1, sharex=True)
        for i, l in enumerate(lun):
            sub=ds3[ds3.ulev == l]
            axs[i].plot(sub.Rt, sub.SumIntensity.astype(float), label='TIC')
            axs[i].plot(sub.Rt, sub.MaxIntensity.astype(float), label='BPC')
            axs[i].set_ylabel(r"$\bfCount$")
            axs[i].legend(loc='best', bbox_to_anchor=(0.5, 0.5, 0.5, 0.5))
            axs[i].ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
            axs[i].set_title('ML_'+'AK_'+'AM_'+'SM_'+'CE_eV: '+l)
        axs[i].text(1.04, 0, self.fname, rotation=90, fontsize=6, transform=axs[i].transAxes)
        fig.canvas.draw()
        axs[i].set_xlabel(r"$\bfScantime (s)$")
    # @log
    @staticmethod
    def csummary(i: int, df):
        import numpy as np
        if df.iloc[:, i].isnull().all():
            return None
        s = np.unique(df.iloc[:, i], return_counts=True)
        n = df.shape[0]
        if len(s[0]) < 5:
            sl = []
            for i in range(len(s[0])):
                sl.append(f'{s[0][i]} ({np.round(s[1][i] / n * 100)}%)')
            return '; '.join(sl)
        else:
            return f'{np.round(np.mean(s), 1)} ({np.round(np.min(s), 1)}-{np.round(np.max(s), 1)})'
    # # @log
    @staticmethod
    def vc(x, n):
        import numpy as np
        ct = x.value_counts(subset=['MsLevel', 'AcqKey', 'ScMode', 'AcqMode'])
        s = []
        for i in range(ct.shape[0]):
            s.append(
                f'{ct.index.names[0]}={ct.index[i][0]} & {ct.index.names[1]}={ct.index[i][1]} &'
                f' {ct.index.names[2]}={ct.index[i][2]} & {ct.index.names[3]}={ct.index[i][3]}: '
                f'{ct[ct.index[i]]} ({np.round(ct[ct.index[i]] / n * 100, 1)} %)')
        return '; '.join(s)


@typechecked
class MSstat:
    def ecdf(self, plot:bool = False):
        # a is the data array
        self.ecdf_x = np.sort(self.Xraw[...,2])
        self.ecdf_y = np.arange(len(self.ecdf_x)) / float(len(self.ecdf_x))
        if plot:
            f, ax = plt.subplots(1,1)
            ax.plot(self.ecdf_x, self.ecdf_y)
            ax.semilogx()
            ax.set_ylabel('p(x_i > X)')
            ax.set_xlabel('x')
        if hasattr(self, 'noise_thres'):
            ax.vlines(x=self.noise_thres, ymin=0, ymax=1, color='red')
    def get_ecdfInt(self, p: float):
        p = np.array(p)[..., np.newaxis]
        if p.ndim ==1:
            p = p[np.newaxis, ...]
        if (any(p < 0)) | (any(p > 1)):
            raise ValueError('p value range: 0-1 ')
        return self.ecdf_x[np.argmin(np.abs(self.ecdf_y - p), axis=1)]
    def calcSignalWindow(self, p: float = 0.999, plot: bool = True, sc: float =300,  **kwargs):
        Xr = self.Xraw[self.Xraw[...,2] > self.get_ecdfInt(p),...]
        Xs=Xr[(Xr[...,1]>300) & (Xr[...,3]<180)]
        t=np.histogram2d(Xs[..., 1]*sc, Xs[..., 3]) #, **kwargs
        extent = [t[2][0], t[2][-1], t[1][0]/sc, t[1][-1]/sc]
        if plot:
            f, axs = plt.subplots(1, 2)
            axs[0].matshow(t[0]/sc,  extent=extent, origin='lower', aspect='auto')
        ii = np.where(t[0] == np.max(t[0]))
        self.mzsignal = mzra = (t[1][ii[0][0]]/sc, t[1][ii[0][0]+1]/sc)
        self.rtsignal = rtra = (t[2][ii[1][0]]-30, t[2][ii[1][0] + 1]+30)
        print(self.rtsignal)
        print(rtra)
        print(f'mz range: {mzra}, rt range (s): {np.round(rtra, 0)}')
        idc=(self.Xraw[..., 1] > mzra[0]) & (self.Xraw[..., 1] < mzra[1]) & (self.Xraw[..., 3] > (rtra[0])) & (self.Xraw[..., 3] < (rtra[1]))
        self.Xsignal = self.Xraw[idc,...]
        self.Xsignal = self.Xsignal[self.Xsignal[...,2]> self.get_ecdfInt(0.99)]
        if plot:
            axs[1].scatter(self.Xsignal[...,3], self.Xsignal[...,1], s=0.5, c=np.log(self.Xsignal[...,2]))
    def scans_sec(self):
        asps = 1/((self.scantime.max() - self.scantime.min())/self.summary.nSc.iloc[0]) # hz
        ii = np.diff(np.unique(self.scantime))
        self.summary = pd.concat([self.summary, pd.DataFrame({'ScansPerSecAver': asps, 'ScansPerSecMin': 1/ii.max(), 'ScansPerSecMax': 1/ii.min()}, index=[0])], axis=1)

@typechecked
class Fdet:
    @log
    def getScaling(self, **kwargs):

        def line_select_callback(eclick, erelease):
            # 'eclick and erelease are the press and release events'
            x1c, y1c = eclick.xdata, eclick.ydata
            x2c, y2c = erelease.xdata, erelease.ydata
            self.dpick=[x1c, y1c, x2c, y2c]
            # print("(%1.2f, %3.2f) --> (%1.2f, %3.2f)" % (x1, y1, x2, y2))
            # print(" The button you used were: %s %s" % (eclick.button, erelease.button))

        def toggle_selector(event):
            pass

        f, ax = self.vis_spectrum1(**kwargs)
        # x1, x2 = ax[0].get_xlim()
        y1, y2 = ax[0].get_ylim()
        # print(x1)
        toggle_selector.RS = RectangleSelector(ax[0], line_select_callback,
                                               drawtype='box', useblit=True,
                                               button=[1, 3],  # don't use middle button
                                               minspanx=5, minspany=5,
                                               spancoords='pixels',
                                               interactive=True)
        ax[0].set_ylim([y1, y2])
        ax[0].callbacks.connect('key_press_event', toggle_selector)
        # f.show()
        #

    @log
    def peak_picking(self, mz_adj: float = 40, q_noise: float = 0.89, dbs_par: dict = {'eps': 1.1, 'min_samples': 3},
                     selection={'mz_min': None, 'mz_max': None, 'rt_min': None, 'rt_max': None},
                     qc_par: dict = {'st_len_min': 5, 'raggedness': 0.15, 'non_neg': 0.8, 'sId_gap': 0.15, 'sino': 2,
                             'ppm': 15}, \
                     qcm_local: bool=True)\
            :

        s0 = time.time()
        self.mz_adj = mz_adj
        self.q_noise = q_noise
        self.dbs_par = dbs_par
        self.qc_par = qc_par
        self.selection = selection

        # if desired reduce mz/rt regions for ppicking
        # Xraw has 4 columns: scanId, mz, int, st
        self.Xf = self.window_mz_rt(self.Xraw, self.selection, allow_none=True, return_idc=False)

        # det noiseInt and append to X
        if qcm_local:
            self.noise_thres = np.quantile(self.Xf[..., 2], q=self.q_noise)
        else:
            self.noise_thres = np.quantile(self.Xraw[..., 2], q=self.q_noise)

        # filter noise data points
        idx_signal = np.where(self.Xf[..., 2] > self.noise_thres)[0]
        noise = np.ones(self.Xf.shape[0])
        noise[idx_signal] = 0
        self.Xf = np.c_[self.Xf, noise]
        # Xf has 5 columns: scanId, mz, int, st, noiseBool
        # st adjustment and append to X
        st_c = np.zeros(self.Xf.shape[0])
        st_c[idx_signal] = self.Xf[idx_signal, 1] * self.mz_adj
        self.Xf = np.c_[self.Xf, st_c]
        # Xf has 6 columns: scanId, mz, int, st, noiseBool, st_adj
        print(f'Clustering, number of dp: {len(idx_signal)}')
        s1 = time.time()
        dbs = DBSCAN(eps=self.dbs_par['eps'], min_samples=self.dbs_par['min_samples'], algorithm='auto').fit(
            self.Xf[idx_signal, :][..., np.array([0, 5])])
        s2 = time.time()

        print(f'dt dbscan: {np.round(s2 - s1)} s')
        # append cluster membership to windowed Xraw
        self.cl_labels = np.unique(dbs.labels_, return_counts=True)
        cl_mem = np.ones(self.Xf.shape[0]) * -1
        cl_mem[idx_signal] = dbs.labels_
        self.Xf = np.c_[self.Xf, cl_mem]  # Xf has 7 columns: scanId, mz, int, st, noiseBool, st_adj, clMem
        s3 = time.time()
        self.feat_summary1()
        s5 = time.time()

        print(f'total time pp (min): {np.round((s5-s0)/60, 1)}')

    @log
    def feat_summary1(self):
        t1 = time.time()
        self.feat = {}
        self.feat_l2 = []
        self.feat_l3 = []
        print(f'Number of L0 features: {len(self.cl_labels[1])}')
        # filter for feature length
        min_le_bool = (self.cl_labels[1] >= self.qc_par['st_len_min']) & (self.cl_labels[0] != -1)
        cl_n_above = self.cl_labels[0][min_le_bool]
        print(f'Number of L1 features: {len(cl_n_above)}')
        self.feat.update(
            {'id:' + str(x): {'flev': 1, 'crit': 'n < st_len_min'} for x in self.cl_labels[0][~min_le_bool]})
        t2 = time.time()
        c = 0
        for fid in cl_n_above:
            c += 1
            f1 = self.cl_summary(fid)
            ufid = 'id:' + str(fid)
            self.feat.update({ufid: f1})
            # print(f1['flev'])
            if f1['flev'] == 2:
                self.feat_l2.append(ufid)
            if f1['flev'] == 3:
                self.feat_l3.append(ufid)
        t3 = time.time()

        qq = [self.feat[x]['qc'][0] if isinstance(self.feat[x]['qc'], type(())) else self.feat[x]['qc'] for x in
              self.feat_l2]

        if len(self.feat_l2) == 0:
            raise ValueError('No L2 features found')
        else:
            print(f'Number of L2 features: {len(self.feat_l2)+len(self.feat_l3)}')

        if len(self.feat_l3) == 0:
            raise ValueError('No L3 features found')
        else:
            print(f'Number of L3 features: {len(self.feat_l3)}')

    @log
    def vis_feature_pp(self, selection: dict = {'mz_min': 425, 'mz_max': 440, 'rt_min': 400, 'rt_max': 440}, lev: int = 3):
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
        fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 2]}, figsize=(8, 10),
                                constrained_layout=False)
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        axs[0].set_ylabel(r"$\bfCount$")
        axs[0].set_xlabel(r"$\bfScan time$, s")

        vis_feature(self.feat[self.feat_l2[0]], id=self.feat_l2[0], ax=axs[0])
        axs[1].set_facecolor('#EBFFFF')
        axs[1].text(1.03, 0, self.fname, rotation=90, fontsize=6, transform=axs[1].transAxes)

        # fill axis for raw data results
        idc_above = np.where(Xsub[:, 2] > self.noise_thres)[0]
        idc_below = np.where(Xsub[:, 2] <= self.noise_thres)[0]

        cm = plt.cm.get_cmap('rainbow')
        axs[1].scatter(Xsub[idc_below, 3], Xsub[idc_below, 1], s=0.1, c='gray', alpha=0.5)

        imax = np.log(Xsub[idc_above, 2])
        psize = (imax/np.max(imax)) * 5

        im = axs[1].scatter(Xsub[idc_above, 3], Xsub[idc_above, 1], c=np.log(Xsub[idc_above, 2]), s=psize, cmap=cm)


        # axs[1].scatter(Xsub[Xsub[:, 4] == 0, 3], Xsub[Xsub[:, 4] == 0, 1], s=0.1, c='gray', alpha=0.5)

        if lev < 3:
            for i in l2_ffeat:
                fid = float(i.split(':')[1])
                f1 = self.feat[i]
                # if (f1['flev'] == 2):
                a_col = 'red'
                fsize = 6
                mz = Xsub[Xsub[:, 6] == fid, 1]
                if len(mz) == len(f1['fdata']['st']):
                    im = axs[1].scatter(f1['fdata']['st'], mz,
                                        c=np.log(f1['fdata']['I_raw']), s=5, cmap=cm)
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
                        im = axs[1].scatter(f1['fdata']['st'], mz,
                                            c=np.log(f1['fdata']['I_raw']), s=5, cmap=cm)
        cbaxes = inset_axes(axs[1], width="30%", height="5%", loc=3)
        plt.colorbar(im, cax=cbaxes, ticks=[0., 1], orientation='horizontal')

        def p_text(event):
            # print(event.artist)
            ids = str(event.artist.get_text())
            if event.mouseevent.button is MouseButton.LEFT:
                # print('left click')
                # print('id:' + str(ids))
                axs[0].clear()
                axs[0].set_title('')
                vis_feature(self.feat['id:' + ids], id=ids, ax=axs[0], add=False)
                event.canvas.draw()
            if event.mouseevent.button is MouseButton.RIGHT:
                # print('right click')
                # print('id:' + str(ids))
                vis_feature(self.feat['id:' + ids], id=ids, ax=axs[0], add=True)
                event.canvas.draw()

        cid1 = fig.canvas.mpl_connect('pick_event', p_text)
        axs[1].set_xlabel(r"$\bfScan time$, s")
        axs[1].yaxis.offsetText.set_visible(False)
        axs[1].yaxis.set_label_text(r"$\bfm/z$")

    @log
    def cl_summary(self, cl_id: Union[int, np.int64]):
        # cl_id = cl_n_above[0]
        # cluster summary
        # X is data matrix
        # dbs is clustering object with lables
        # cl_id is cluster id in labels
        # st_adjust is scantime adjustment factor

        # Xf has 7 columns: scanId, mz, int, st, noiseBool, st_adj, clMem


        # import matplotlib.pyplot as plt
        # cl_id = 3660
        @log
        def wImean(x, I):
            # intensity weighted mean of st
            w = I / np.sum(I)
            wMean_val = np.sum(w * x)
            wMean_idx = np.argmin(np.abs(x - wMean_val))
            return (wMean_val, wMean_idx)

        @log
        def ppm_mz(mz):
            return np.std(mz) / np.mean(mz) * 1e6

        # dp idx for cluster
        idx = np.where(self.Xf[:, 6] == cl_id)[0]
        iLen = len(idx)

        # gaps or doublets in successive scan ids
        x_sid = self.Xf[idx, 0]
        sid_diff = np.diff(x_sid)
        sid_gap = np.sum((sid_diff > 1) | (sid_diff < 1)) / (iLen - 1)

        # import matplotlib.pyplot as plt
        x = self.Xf[idx, 3]  # scantime
        y = self.Xf[idx, 2]  # intensity
        mz = self.Xf[idx, 1]  # mz
        ppm = ppm_mz(mz)

        mz_min = np.min(mz)
        mz_max = np.max(mz)
        rt_min = np.min(x)
        rt_max = np.max(x)

        if ppm > self.qc_par['ppm']:
            crit = 'm/z variability of a feature (ppm)'
            qc = {'sId_gap': sid_gap, 'ppm': ppm}
            quant = {}
            descr = {'ndp': iLen, 'st_span': np.round(rt_max - rt_min, 1), \
                     'mz_min': mz_min, 'mz_max': mz_max, \
                     'rt_min': rt_min, 'rt_max': rt_max, }
            fdata = {'st': x, 'I_raw': y}
            od = {'flev': 2, 'crit': crit, 'qc': qc, 'quant': quant, 'descr': descr, 'fdata': fdata}
            return od

        if sid_gap > self.qc_par['sId_gap']:
            crit = 'sId_gap: scan ids not consecutive (%)'
            qc = {'sId_gap': sid_gap, 'ppm': ppm}
            quant = {}
            descr = {'ndp': iLen, 'st_span': np.round(rt_max - rt_min, 1), \
                     'mz_min': mz_min, 'mz_max': mz_max, \
                     'rt_min': rt_min, 'rt_max': rt_max, }
            fdata = {'st': x, 'I_raw': y}
            od = {'flev': 2, 'crit': crit, 'qc': qc, 'quant': quant, 'descr': descr, 'fdata': fdata}
            return od

        _, icent = wImean(x, y)

        # minor smoothing - minimum three data points
        ysm = np.convolve(y, np.ones(3) / 3, mode='valid')
        xsm = x[1:-1]

        # padding
        ysm = np.concatenate(([y[0]], ysm, [y[-1]]))
        xsm = x
        idxsm = idx[1:-1]

        # I min-max scaling
        y_max = np.max(y)
        y = y / y_max
        ysm_max = np.max(ysm)
        ysm = ysm / ysm_max

        # bline correction of smoothed signal
        # if smoothed signal has less than three data points, then this won't work
        # minimum last points minu
        y_bl = ((np.min(ysm[-3:]) - np.min(ysm[0:3])) / (xsm[-1] - xsm[0])) * (xsm - np.min(xsm)) + np.min(ysm[0:3])

        ybcor = ysm - y_bl
        bcor_max = np.max(ybcor * ysm_max)

        # check peak centre
        _, icent = wImean(xsm, ybcor)

        # if (icent < 2) | (icent > len(idxsm) - 2):
        if (icent < 2) | (icent > (len(idx) - 2)):
            crit = 'icent: Peak is not centered'
            qc = {'sId_gap': sid_gap, 'icent': icent, 'ppm': ppm}
            quant = {}
            descr = {'ndp': iLen, 'st_span': np.round(rt_max - rt_min, 1), \
                     'mz_min': mz_min, 'mz_max': mz_max, \
                     'rt_min': rt_min, 'rt_max': rt_max, }
            fdata = {'st': x, 'st_sm': xsm, 'I_sm_bline': ybcor * ysm_max, 'I_smooth': ysm * ysm_max,
                     'I_raw': y * y_max,
                     'I_bl': y_bl * ysm_max, }
            od = {'flev': 2, 'crit': crit, 'qc': qc, 'quant': quant, 'descr': descr, 'fdata': fdata}
            return od

        nneg = np.sum(ybcor >= -0.05) / len(ysm)
        if nneg < self.qc_par['non_neg']:
            crit = 'nneg: Neg Intensity values'
            qc = {'sId_gap': sid_gap, 'non_neg': nneg, 'ppm': ppm}
            quant = {}
            descr = {'ndp': iLen, 'st_span': np.round(rt_max - rt_min, 1), \
                     'mz_min': mz_min, 'mz_max': mz_max, \
                     'rt_min': rt_min, 'rt_max': rt_max, }
            fdata = {'st': x, 'st_sm': xsm, 'I_sm_bline': ybcor * ysm_max, 'I_smooth': ysm * ysm_max,
                     'I_raw': y * y_max, 'I_bl': y_bl * ysm_max, }
            od = {'flev': 2, 'crit': crit, 'qc': qc, 'quant': quant, 'descr': descr, 'fdata': fdata}
            return od

        # signal / noise
        scans = np.unique(x_sid)
        scan_id = np.concatenate([np.where((self.Xf[:, 0] == x) & (self.Xf[:, 4] == 1))[0] for x in scans])
        noiI = np.median(self.Xf[scan_id, 2])
        sino = bcor_max / noiI

        if sino < self.qc_par['sino']:
            crit = 'sino: s/n below qc threshold'
            qc = {'sId_gap': sid_gap, 'non_neg': nneg, 'sino': sino, 'ppm': ppm}
            quant = {}
            descr = {'ndp': iLen, 'st_span': np.round(rt_max - rt_min, 1), \
                     'mz_min': mz_min, 'mz_max': mz_max, \
                     'rt_min': rt_min, 'rt_max': rt_max, }
            fdata = {'st': x, 'st_sm': xsm, 'I_sm_bline': ybcor * ysm_max, 'I_smooth': ysm * ysm_max,
                     'I_raw': y * y_max,
                     'I_bl': y_bl * ysm_max, }
            od = {'flev': 2, 'crit': crit, 'qc': qc, 'quant': quant, 'descr': descr, 'fdata': fdata}
            return od

        # intensity variation level
        idx_maxI = np.argmax(ybcor)
        sdxdy = (np.sum(np.diff(ybcor[:idx_maxI]) < (-0.05)) + np.sum(np.diff(ybcor[(idx_maxI + 1):]) > 0.05)) / (
                len(ybcor) - 3)

        if sdxdy > self.qc_par['raggedness']:
            crit = 'raggedness: intensity variation level above qc threshold'
            qc = {'sId_gap': sid_gap, 'non_neg': nneg, 'sino': sino, 'raggedness': sdxdy, 'ppm': ppm}
            quant = {}
            descr = {'ndp': iLen, 'st_span': np.round(rt_max - rt_min, 1), \
                     'mz_min': mz_min, 'mz_max': mz_max, \
                     'rt_min': rt_min, 'rt_max': rt_max, }
            fdata = {'st': x, 'st_sm': xsm, 'I_sm_bline': ybcor * ysm_max, 'I_smooth': ysm * ysm_max,
                     'I_raw': y * y_max,
                     'I_bl': y_bl * ysm_max, }
            od = {'flev': 2, 'crit': crit, 'qc': qc, 'quant': quant, 'descr': descr, 'fdata': fdata}
            return od

        # general feature descriptors
        mz_maxI = np.max(self.Xf[idx, 2])
        mz_mean = np.mean(mz)

        rt_maxI = x[np.argmax(self.Xf[idx, 2])]
        rt_mean = np.mean(xsm)

        # create subset with equal endings
        _, icent = wImean(xsm, ybcor)

        # tailing
        tail = rt_maxI / rt_mean

        # d = np.min([icent, np.abs(icent - len(idxsm))])
        # ra=range((icent-d), (icent+d))

        # xr=xsm[ra]
        # yr=ybcor[ra]
        # yr_max = np.max(yr)
        # yr =  yr / yr_max
        a = integrate.trapz(ybcor * bcor_max, x=xsm)
        a_raw = integrate.trapz(y, x)
        a_sm = integrate.trapz(ysm, xsm)

        # I mind not need this here
        # include symmetry and monotonicity
        pl = find_peaks(ybcor * bcor_max, distance=10)
        plle = len(pl[0])

        if plle > 0:
            pp = peak_prominences(ybcor * bcor_max, pl[0])
            if len(pp[0]) > 0:
                aa = str(np.round(pp[0][0]))
            else:
                aa = 0
        else:
            pl = -1
            aa = -1
        qc = {'sId_gap': sid_gap, 'non_neg': nneg, 'sino': sino, 'raggedness': sdxdy, 'tail': tail, 'ppm': ppm}
        quant = {'raw': a_raw, 'sm': a_sm, 'smbl': a}
        descr = {'ndp': iLen, 'npeaks': plle, 'pprom': aa, 'st_span': np.round(rt_max - rt_min, 1), \
                 'mz_maxI': mz_maxI, 'rt_maxI': rt_maxI, \
                 'mz_mean': mz_mean, 'mz_min': mz_min, 'mz_max': mz_max, \
                 'rt_mean': rt_mean, 'rt_min': rt_min, 'rt_max': rt_max, }
        fdata = {'st': x, 'st_sm': xsm, 'I_sm_bline': ybcor * ysm_max, 'I_smooth': ysm * ysm_max, 'I_raw': y * y_max,
                 'I_bl': y_bl * ysm_max, }
        od = {'flev': 3, 'qc': qc, 'quant': quant, 'descr': descr, 'fdata': fdata}
        return od

    @log
    def l3toDf(self):
        if len(self.feat_l3) > 0:
            descr = pd.DataFrame([self.feat[x]['descr'] for x in self.feat_l3])
            descr['id'] = self.feat_l3
            quant = pd.DataFrame([self.feat[x]['quant'] for x in self.feat_l3])
            qc = pd.DataFrame([self.feat[x]['qc'] for x in self.feat_l3])

            out =  pd.concat((descr[['mz_mean', 'rt_maxI', 'rt_min', 'rt_max', 'npeaks']], qc[['ppm', 'sino']], quant['smbl']), axis=1)
            out.index = descr['id']
            self.l3Df = out.sort_values('smbl', ascending=False)
            #return self.l3Df
        else:
            print('No L3 feat')

    @log
    def l3gr(self):
        # L3 feature grouping
        self.l3toDf()
        A = self.l3Df.index
        isop = np.zeros(self.l3Df.shape[0])
        count_isop=0
        while len(A) > 0:
            id= self.l3Df.loc[A[0]]
            idc = np.where(((id.rt_maxI - self.l3Df.rt_maxI).abs() < 1) & ((id.mz_mean - self.l3Df.mz_mean) < 4.2))[0]
            isop[idc]=count_isop
            count_isop += 1
            A = list(set(A) - set(self.l3Df.index[idc]))
        self.l3Df['isot'] = isop
        self.l3Df = self.l3Df.sort_values('isot', ascending=True)

@typechecked
class MSexp(Fdet, MSstat):
    @log
    def __init__(self, dpath: str, convert:bool = True, docker:dict={'repo': 'convo:v1', 'mslevel': 0, 'smode': [0, 1, 2, 3, 4, 5], 'amode': 2 , 'seg': [0, 1, 2, 3],} ): #  mslev='1', print_summary=False,
        self.mslevel = str(docker['mslevel']) if not isinstance(docker['mslevel'], list) else tuple(map(str, np.unique(docker['mslevel'])))
        if len(self.mslevel) == 1:
            self.mslevel = self.mslevel[0]
        self.smode = str(docker['smode']) if not isinstance(docker['smode'], list) else tuple(map(str, np.unique(docker['smode'])))
        if len(self.smode) == 1:
            self.smode = self.smode[0]
        self.amode = str(docker['amode'])
        self.seg = str(docker['seg']) if not isinstance(docker['seg'], list) else tuple(map(str, np.unique(docker['seg'])))
        if len(self.seg) == 1:
            self.seg = self.seg[0]
        self.docker = {'repo': docker['repo'], 'mslevel': self.mslevel , 'smode': self.smode, 'amode': self.amode , 'seg': self.seg,}
        self.dpath = os.path.abspath(dpath)
        self.dbfile = os.path.join(dpath, 'analysis.sqlite')
        self.fname = os.path.basename(dpath)
        # does it contain msmate object
        self.msmfile = os.path.join(dpath,
                     f'mm8v3_edata_msl{self.docker["mslevel"]}_sm{"".join(self.docker["smode"])}_am{self.docker["amode"]}_seg{"".join(self.docker["seg"])}.p')
        if os.path.exists(self.msmfile):
            self.read_mm8()
        else:
            if convert:
                self.convoDocker()
                self.read_mm8()
            else:
                raise SystemError('d folder requries conversion, enable docker convo')
        self.ecdf()
    @log
    def convoDocker(self):
        t0=time.time()
        client = dock.from_env()
        client.info()
        client.containers.list()
        # img = [x for x in client.images.list() if self.docker["repo"] in x.tags]
        img = [x for x in client.images.list() if self.docker["repo"] in x.tags]
        if len(img) < 1:
            raise ValueError('Image not found')
        ec = f'docker run -v "{os.path.dirname(self.dpath)}":/data {self.docker["repo"]} "edata" "{self.fname}" -l {self.docker["mslevel"]} -am {self.docker["amode"]} -sm {" ".join(self.docker["smode"])} -s {" ".join(self.docker["seg"])}'
        print(ec)
        os.system(ec)
        t1 = time.time()
        print(f'Conversion time: {np.round(t1 - t0)} sec')

    def msZeroPP(self):
        if any((self.df.ScanMode == 2) & ~(self.df.MsLevel == 1)):
            raise ValueError('Check AcquisitionKeys for scan types - combo not encountered before')

        if any((self.df.ScanMode == 5) & ~(self.df.MsLevel == 0)):
            raise ValueError('Check AcquisitionKeys for scan types - combo not encountered before')


        idx_dda = (self.df.ScanMode == 2) & (self.df.ScanMode == 1)
        if any(idx_dda):
            print('DDA')
        idx_dia = (self.df.ScanMode == 5) & (self.df.MsLevel == 0)
        if any(idx_dia):
            print('DIA')
        idx_ms0 = (self.df.ScanMode == 0) & (self.df.MsLevel == 0)

        self.df['LevPP'] = None
        add =  self.df['LevPP'].copy()
        add.loc[idx_dda] = 'dda'
        add.loc[idx_dia] = 'dia'
        add.loc[idx_ms0] = 'fs'

        self.df['LevPP'] = add

    @log
    def read_mm8(self):
        ss = pickle.load(open(self.msmfile, 'rb'))

        self.df = pd.DataFrame(ss['sinfo'])
        self.msZeroPP()

        # define ms level for viz and peak picking


        idcS3 = (self.df.Segment == 3).values
        if any(idcS3):
            self.df['stype'] = "0"
            add = self.df['stype'].copy()
            add[idcS3] = [f'{x["MsLevel"]}_{x["AcquisitionKey"]}_{x["AcquisitionMode"]}_{x["ScanMode"]}_{x["Collision_Energy_Act"]}' for x in ss['sinfo'] if x['Segment'] == 3]
            self.df['stype'] = add

        self.scanid = np.concatenate([np.ones_like(ss['LineIndexId'][i][1]) * i for i in range(len(ss['sinfo']))]).astype(int)
        self.mz = np.concatenate([x[1] for x in ss['LineMzId']])
        self.I = np.concatenate([x[1] for x in ss['LineIntensityId']])
        self.scantime = np.concatenate([np.ones_like(ss['LineIndexId'][i][1]) * ss['sinfo'][i]['Rt'] for i in range(len(ss['sinfo']))])
        self.nd = len(self.I)
        self.Xraw = np.array([self.scanid, self.mz, self.I, self.scantime]).T


        polmap = {0: 'P', 1: 'N'}
        df1 = pd.DataFrame(
            [(polmap[x['Polarity']], str(x['MzAcqRangeLower']) + '-' + str(x['MzAcqRangeUpper']), x['MsLevel']) for x in ss['sinfo']], columns=['Polarity', 'MzAcqRange', 'MsLevel'])
        df1 = pd.DataFrame([self.csummary(i, df1) for i in range(df1.shape[1])], index=df1.columns).transpose()
        df1['nSc'] = len(ss['sinfo'])

        df2 = pd.DataFrame([(x['AcquisitionMode'], x['ScanMode'], x['Segment']) for x in ss['sinfo']],
                           columns=['AcqMode', 'ScMode', 'Segments'])
        t = pd.DataFrame(df2.groupby(['Segments']).apply(lambda x: self.vc(x, n=df2.shape[0])))
        t = t.rename(index={t.index[i]: 'Segment ' + str(t.index[i]) for i in range(t.shape[0])}).transpose()
        asps = 1 / ((self.scantime.max() - self.scantime.min()) /  df1['nSc'].iloc[0])  # hz
        ii = np.diff(np.unique(self.scantime))
        sps = pd.DataFrame(
            {'ScansPerSecAver': asps, 'ScansPerSecMin': 1 / ii.max(), 'ScansPerSecMax': 1 / ii.min()}, index=[0])
        self.summary = pd.concat([df1, t, sps], axis=1)

    # @log
    @staticmethod
    def csummary(i: int, df: pd.DataFrame):
        if df.iloc[:, i].isnull().all():
            return None
        s = np.unique(df.iloc[:, i], return_counts=True)
        n = df.shape[0]
        if len(s[0]) < 5:
            sl = []
            for i in range(len(s[0])):
                sl.append(f'{s[0][i]} ({np.round(s[1][i] / n * 100)}%)')
            return '; '.join(sl)
        else:
            return f'{np.round(np.mean(s), 1)} ({np.round(np.min(s), 1)}-{np.round(np.max(s), 1)})'
    # @log
    @staticmethod
    def vc(x, n):
        import numpy as np
        ct = x.value_counts(subset=['ScMode', 'AcqMode'])
        s = []
        for i in range(ct.shape[0]):
            s.append(
                f'{ct.index.names[0]}={ct.index[i][0]} & {ct.index.names[1]}={ct.index[i][1]}: {ct[ct.index[i]]} ({np.round(ct[ct.index[0]] / n * 100, 1)} %)')
        return '; '.join(s)
    # @log
    @staticmethod
    def window_mz_rt(Xr: np.ndarray, selection:dict ={'mz_min': None, 'mz_max': None, 'rt_min': None, 'rt_max': None},
                     allow_none:bool=False, return_idc:bool=False):
        # make selection on Xr based on rt and mz range
        # allow_none: true if None alowed in selection (then full range is used)
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
    @log
    def vis_spectrum(self, q_noise:float=0.89, selection:dict={'mz_min': None, 'mz_max': None, 'rt_min': None, 'rt_max': None}):
        # visualise part of the spectrum, before peak picking

        Xsub = self.window_mz_rt(self.Xraw, selection, allow_none=False)

        # filter noise points
        noise_thres = np.quantile(self.Xraw[..., 2], q=q_noise)
        idc_above = np.where(Xsub[:, 2] > noise_thres)[0]
        idc_below = np.where(Xsub[:, 2] <= noise_thres)[0]

        cm = plt.cm.get_cmap('rainbow')
        # fig, ax = plt.subplots()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.text(1.25, 0, self.fname, rotation=90, fontsize=6, transform=ax.transAxes)

        ax.scatter(Xsub[idc_below, 3], Xsub[idc_below, 1], s=0.1, c='gray', alpha=0.5)
        im = ax.scatter(Xsub[idc_above, 3], Xsub[idc_above, 1], c=np.log(Xsub[idc_above, 2]), s=5, cmap=cm)
        fig.canvas.draw()

        ax.set_xlabel(r"$\bfScan time$ (sec)")
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
        ax2.set_xlabel(r"Scan time (min)")
        fig.colorbar(im, ax=ax)
        fig.show()

        return (fig, ax)
    # @log
    @staticmethod
    def get_density(X:np.ndarray, bw:float, q_noise:float=0.5):
        xmin = X[:, 1].min()
        xmax = X[:, 1].max()
        b = np.linspace(xmin, xmax, 500)
        idxf = np.where(X[:, 2] > np.quantile(X[:, 2], q_noise))[0]
        x_rev = X[idxf, 1]
        kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(x_rev[:, np.newaxis])
        log_dens = np.exp(kde.score_samples(b[:, np.newaxis]))
        return (b, log_dens / np.max(log_dens))
    @log
    def vis_spectrum1(self, q_noise:float=0.50, selection:dict={'mz_min': None, 'mz_max': None, 'rt_min': None, 'rt_max': None}, qcm_local: bool=False):
        @log
        def on_lims_change(event_ax: plt.axis):  # f=self.get_density, fil = self.window_mz_rt
            # print(event_ax.__class__)
            x1, x2 = event_ax.get_xlim()
            y1, y2 = event_ax.get_ylim()
            selection = {'mz_min': y1, 'mz_max': y2, 'rt_min': x1, 'rt_max': x2}
            Xs = self.window_mz_rt(self.Xraw, selection, allow_none=False)
            bsMin = (y2 - y1) / 200 if ((y2 - y1) / 200) > 0.0001 else 0.0001
            bw = bsMin
            y, x = self.get_density(Xs, bw, q_noise=q_noise)

            axs[1].clear()
            axs[1].set_ylim([y1, y2])
            axs[1].plot(x, y, c='black')
            axs[1].tick_params(labelleft=False)
            axs[1].set_xticks([])
            axs[1].annotate(f'bw: {np.round(bw, 5)}\np: {np.round(q_noise, 3)}',
                            xy=(1, y1 + np.min([0.1, float((y2 - y1) / 100)])), fontsize='xx-small', ha='right')
            axs[1].text(1.05, 0, self.fname, rotation=90, fontsize=4, transform=axs[1].transAxes)
            axs[1].tick_params(axis='y', direction='in')

        idl0 = (self.df.Id[self.df['LevPP'] == 'fs']-1).values
        [x in idl0 for x in self.Xraw[:,0]]

        self.df.Id[self.df['LevPP'] == 'fs']-1
        Xsub = self.window_mz_rt(self.Xraw[self.df['LevPP'] == 'fs'], selection, allow_none=False)
        if qcm_local:
            noise_thres, vmi, vma = np.quantile(Xsub[..., 2], q=[q_noise, 0, 1])
        else:
            noise_thres, vmi, vma = np.quantile(self.Xraw[self.df['LevPP'] == 'fs', 2], q=[q_noise, 0, 1])
        idc_above = np.where(Xsub[:, 2] > noise_thres)[0]
        idc_below = np.where(Xsub[:, 2] <= noise_thres)[0]
        cm = plt.cm.get_cmap('rainbow')
        fig, axs = plt.subplots(1, 2, sharey=False, gridspec_kw={'width_ratios': [3, 1]})
        axs[1].text(1.05, 0, self.fname, rotation=90, fontsize=4, transform=axs[1].transAxes)
        axs[1].tick_params(axis='y', direction='in')
        axs[0].scatter(Xsub[idc_below, 3], Xsub[idc_below, 1], s=0.1, c='gray', alpha=0.5)
        im = axs[0].scatter(Xsub[idc_above, 3], Xsub[idc_above, 1], c=(Xsub[idc_above, 2]), s=5, cmap=cm,
                            norm=LogNorm(vmi, vma))
        cbaxes = fig.add_axes([0.15, 0.77, 0.021, 0.1])
        cb = fig.colorbar(mappable=im, cax=cbaxes, orientation='vertical',
                          format=LogFormatterSciNotation(base=10, labelOnlyBase=False))
        cb.ax.tick_params(labelsize='xx-small')
        fig.subplots_adjust(wspace=0.05)
        axs[0].callbacks.connect('ylim_changed', on_lims_change)
        return (fig, axs)

    def massSpectrumDIA(self, sid=30, scantype='0_2_2_5_20.0'):
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        import numpy as np

        idc = np.where(self.df.stype == scantype)[0]
        idx = np.argmin(np.abs((self.df.Id.iloc[idc]-1)-sid))
        inf = self.df.iloc[idc[idx]]

        sub = self.Xraw[self.Xraw[:,0] == inf.Id-1]
        f, ax = plt.subplots(1, 1)
        mz = sub[:,1]
        intens =  sub[:,2]
        ax.vlines(sub[:,1], np.zeros_like(mz), intens/np.max(intens) *100)
        ax.text(0.77, 0.95, f'Acq Type: {inf.LevPP.upper()}', rotation=0, fontsize=8, transform=ax.transAxes, ha='left')
        ax.text(0.77, 0.9, f'Rt: {inf.Rt} s', rotation=0, fontsize=8, transform=ax.transAxes, ha='left')
        ax.text(0.77, 0.85, f'Collision E: {inf.Collision_Energy_Act} eV', rotation=0, fontsize=8, transform=ax.transAxes, ha='left')
        return (f, ax)

    def ms2L(self, q_noise, selection):
        pass

    # def vis_spectrum2(self, q_noise=0.50, selection={'mz_min': None, 'mz_max': None, 'rt_min': None, 'rt_max': None}):
    #     def on_lims_change(event_ax):
    #         x1, x2 = event_ax.get_xlim()
    #         y1, y2 = event_ax.get_ylim()
    #         selection = {'mz_min': y1, 'mz_max': y2, 'rt_min': x1, 'rt_max': x2}
    #
    #         Xs = self.window_mz_rt(self.Xraw, selection, allow_none=False)
    #         print(Xs.shape)
    #         scsum = []
    #         for i in np.unique(Xs[:,0]):
    #             scsum.append(np.sum(Xs[Xs[:,0] == i,2]))
    #         bsMin = (y2 - y1) / 200 if ((y2 - y1) / 200) > 0.0001 else 0.0001
    #         bw = bsMin
    #         y, x = self.get_density(Xs, bw, q_noise=q_noise)
    #
    #         axs[1, 1].clear()
    #         axs[1, 1].set_ylim([y1, y2])
    #         axs[1, 1].plot(x, y, c='black')
    #         axs[1, 1].tick_params(labelleft=False)
    #         axs[1, 1].set_xticks([])
    #         axs[1, 1].annotate(f'bw: {np.round(bw, 5)}\np: {np.round(q_noise, 3)}',
    #                         xy=(1, y1 + np.min([0.1, float((y2 - y1) / 100)])), fontsize='xx-small', ha='right')
    #         axs[1, 1].text(1.05, 0, self.fname, rotation=90, fontsize=4, transform=axs[1,1].transAxes)
    #         axs[1, 1].tick_params(axis='y', direction='in')
    #
    #         axs[0,0].clear()
    #         axs[0,0].set_xlim([x1, x2])
    #         #axs[0,0].set_ylim([])
    #         st = np.unique(Xs[:, 3])
    #         print(st)
    #         plt.figure()
    #         plt.plot(st, scsh )
    #         axs[0,0].plot(st, scsum, c='black')
    #         axs[0,0].tick_params(labelbottom=False, axis='x', direction='in')
    #         axs[0,0].set_yticks([])
    #         # axs[0,0].annotate(f'bw: {np.round(bw, 5)}\np: {np.round(q_noise, 3)}',
    #         #                    xy=(1, y1 + np.min([0.1, float((y2 - y1) / 100)])), fontsize='xx-small', ha='right')
    #         # axs[0,0].text(1.05, 0, self.fname, rotation=90, fontsize=4, transform=axs[1].transAxes)
    #         #axs[0,0].tick_params(axis='y', direction='in')
    #
    #
    #
    #
    #
    #     import matplotlib.pyplot as plt
    #     from matplotlib.colors import LogNorm
    #     from matplotlib.ticker import LogFormatterSciNotation
    #     import numpy as np
    #     Xsub = self.window_mz_rt(self.Xraw, selection, allow_none=False)
    #     noise_thres = np.quantile(self.Xraw[..., 2], q=q_noise)
    #     idc_above = np.where(Xsub[:, 2] > noise_thres)[0]
    #     idc_below = np.where(Xsub[:, 2] <= noise_thres)[0]
    #     cm = plt.cm.get_cmap('rainbow')
    #     fig, axs = plt.subplots(2, 2, sharey=False, gridspec_kw={'width_ratios': [3, 0.7], 'height_ratios': [0.7, 3]})
    #     axs[1, 1].text(1.05, 0, self.fname, rotation=90, fontsize=4, transform=axs[1, 1].transAxes)
    #     axs[1, 1].tick_params(axis='y', direction='in')
    #     #axs[0, 1].tick_params(axis='y', direction='in')
    #     axs[0, 0].tick_params(axis='both', direction='in')
    #     axs[1, 0].scatter(Xsub[idc_below, 3], Xsub[idc_below, 1], s=0.1, c='gray', alpha=0.5)
    #     im = axs[1, 0].scatter(Xsub[idc_above, 3], Xsub[idc_above, 1], c=(Xsub[idc_above, 2]), s=5, cmap=cm,
    #                         norm=LogNorm())
    #     cbaxes = fig.add_axes([0.15, 0.57, 0.021, 0.1])
    #     cb = fig.colorbar(mappable=im, cax=cbaxes, orientation='vertical',
    #                       format=LogFormatterSciNotation(base=10, labelOnlyBase=False))
    #     cb.ax.tick_params(labelsize='xx-small')
    #     fig.subplots_adjust(wspace=0.05, hspace=0.05)
    #     axs[1, 0].callbacks.connect('ylim_changed', on_lims_change)
    @log
    def plt_chromatogram(self, tmin:float=None, tmax:float=None, ctype:list =['tic', 'bpc', 'xic'], xic_mz:float =[], xic_ppm:float=10):

        df_l1 = self.df[self.df['LevPP'] == 'fs']
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
            tmin = df_l1.Rt.min()
        if isinstance(tmax, type(None)):
            tmax = df_l1.Rt.max()
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        if xic:
            xx, xy = self.xic(xic_mz, xic_ppm, rt_min=tmin, rt_max=tmax)
            ax1.plot(xx, xy.astype(float), label='XIC ' + str(xic_mz) + 'm/z')
        if tic:
            ax1.plot(df_l1.Rt, df_l1.SumIntensity.astype(float), label='TIC')
        if bpc:
            ax1.plot(df_l1.Rt, df_l1.MaxIntensity.astype(float), label='BPC')
        ax1.text(1.04, 0, self.fname, rotation=90, fontsize=6, transform=ax1.transAxes)
        fig.canvas.draw()
        offset = ax1.yaxis.get_major_formatter().get_offset()
        ax1.set_xlabel(r"$\bfScantime (s)$")
        ax1.set_ylabel(r"$\bfCount$")
        ax2 = ax1.twiny()
        ax1.yaxis.offsetText.set_visible(False)
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
        ax2.set_xlabel(r"$\bfScantime (min)$")
        # ax2.yaxis.set_label_text(r"$Int/count$ " + offset)

        ax1.legend(loc='best', bbox_to_anchor=(0.5, 0.5, 0.5, 0.5))

        fig.show()
        return fig, ax1, ax2


    @log
    def d_ppm(self, mz:float, ppm:float):
        d = (ppm * mz)/ 1e6
        return mz-(d/2), mz+(d/2)

    @log
    def xic(self, mz:float, ppm:float, rt_min:float=None, rt_max:float=None):
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