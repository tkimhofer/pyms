import asyncio

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

import matplotlib as mpl
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

import asyncio
import platform
import os
import psutil



logging.basicConfig(level=logging.DEBUG, filename='example.log', encoding='utf-8', format='%(asctime)s %(levelname)s %(name)s %(message)s', filemode='w')
logger = logging.getLogger(__name__)
logger.debug(f'{[os.cpu_count(), platform.uname(), psutil.net_if_addrs(), psutil.users()]}')


def log(f):
    @functools.wraps(f)
    def wr(*arg, **kwargs):
        try:
            logger.info(f'calling {f.__name__}')
            out = f(*arg, **kwargs)
            logger.info(f'done {f.__name__}')
            return out
        except Exception as e:
            logger.exception(f'{f.__name__}: {repr(e)}')
    return wr



def logIA(f):
    @functools.wraps(f)
    def wr(*arg, **kwargs):
        try:
            logger.info(f'calling {f.__name__}')
            logger.info(f'input args {f.__dict__}')
            out = f(*arg, **kwargs)
            logger.info(f'done {f.__name__}')
            return out
        except Exception as e:
            logger.exception(f'{f.__name__}: {repr(e)}')
    return wr


#
# def background(f):
#     def wr(*args, **kwargs):
#         return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)
#     return wr
# #
#
# def keggMass(m='180', names=True):
#     import requests
#     q=f'https://rest.kegg.jp/find/compound/{m}/exact_mass'
#     r = requests.get(q)
#     if r.status_code == 200:
#         cids = [x.split('\t') for x in r.text.split('\n') if not x in '']
#         print(f'# isobaric compounds on kegg {len(cids)}')
#
#         if names:
#             s = [requests.get(f'https://rest.kegg.jp/get/{c.split(":")[1]}/') for c, w in cids]
#             out = [x.text.split('\n')[1].split(' ')[-1].split(';')[0] for x in s]
#             outd = {cids[i][0]: (o, cids[i][1]) for i, o in enumerate(out)}
#         else:
#             outd: len(cids)
#     else:
#         outd = None
#     return outd

@logIA
@typechecked
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
    def chromg(self, ds3):
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

    @staticmethod
    def assignInstrumentLabel():

        ilab = {'1825265.10252': ('PAI01', 'impact II'),

                '1825265.10251': ('PAI03', 'impact II'),
                '1825265.10274': ('PAI04', 'impact II'),
                '1825265.10271': ('PAI05', 'impact II'),
                '1825265.10258': ('PAI06', 'impact II'),
                '1825265.10267': ('PAI07', 'impact II'),
                '1825265.10269': ('PAI08', 'impact II'),

                '1825265.10275': ('RAI02', 'impact II'),
                '1825265.10273': ('RAI03', 'impact II'),
                '1825265.10272': ('RAI04', 'impact II')}

        # ('PAT01', None, 'Waters')
        # ('PAT02', None, 'Waters')
        # ('PLIP01', None, 'Sciex')
        # ('RAI01', None, 'Waters')
        # ('RAT01', None, 'Waters')
        # ('REIMS01', None, 'Waters')
        # ('TIMS01', None, 'Bruker')  # no sqlite3 (Jul 22)
        # ('TIMS02', None, 'Bruker')  # no sqlite3 (Jul 22)
        # ('EVOQ', None, 'Bruker TQ')  # no sqlite3 (Jul 22)

@logIA
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
    # def scans_sec(self):
    #     asps = 1/((self.scantime.max() - self.scantime.min())/self.summary.nSc.iloc[0]) # hz
    #     ii = np.diff(np.unique(self.scantime))
    #     self.summary = pd.concat([self.summary, pd.DataFrame({'ScansPerSecAver': asps, 'ScansPerSecMin': 1/ii.max(), 'ScansPerSecMax': 1/ii.min()}, index=[0])], axis=1)

@log
@typechecked
class Fdet:
    @log
    def getScaling(self, **kwargs):
        def line_select_callback(eclick, erelease):
            x1c, y1c = eclick.xdata, eclick.ydata
            x2c, y2c = erelease.xdata, erelease.ydata
            self.dpick=[x1c, y1c, x2c, y2c]
        def toggle_selector(event):
            pass
        f, ax = self.vis_spectrum1(**kwargs)
        y1, y2 = ax[0].get_ylim()
        toggle_selector.RS = RectangleSelector(ax[0], line_select_callback,
                                               drawtype='box', useblit=True,
                                               button=[1, 3],  # don't use middle button
                                               minspanx=5, minspany=5,
                                               spancoords='pixels',
                                               interactive=True)
        ax[0].set_ylim([y1, y2])
        ax[0].callbacks.connect('key_press_event', toggle_selector)

    @log
    def pp(self, mz_adj: float = 40, q_noise: float = 0.99, dbs_par: dict = {'eps': 1.1, 'min_samples': 3},
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
        # Xraw has 5 rows: scanIdOri, mz, int, st, scanIdNorm
        self.Xf = self.window_mz_rt(self.xrawd[self.ms0string], self.selection, allow_none=True, return_idc=False)

        # det noiseInt and append to X
        if qcm_local:
            self.noise_thres = np.quantile(self.Xf[2], q=self.q_noise)
        else:
            self.noise_thres = np.quantile(self.xrawd[self.ms0string][2], q=self.q_noise)

        # filter noise data points
        idx_signal = np.where(self.Xf[2] > self.noise_thres)[0]
        noise = np.ones(self.Xf.shape[1])
        noise[idx_signal] = 0
        self.Xf = np.r_[self.Xf, noise[np.newaxis, ...]]
        # Xf has 6 rows: scanIdOri, mz, int, st, scanIdNorm, noiseBool
        # st adjustment and append to X
        st_c = np.zeros(self.Xf.shape[1])
        st_c[idx_signal] = self.Xf[1, idx_signal] * self.mz_adj
        self.Xf = np.r_[self.Xf, st_c[np.newaxis, ...]]
        # Xf has 7 rows: scanIdOri, mz, int, st, scanIdNorm, noiseBool, st_adj
        print(f'Number of dp: {len(idx_signal)}')
        s1 = time.time()
        dbs = DBSCAN(eps=self.dbs_par['eps'], min_samples=self.dbs_par['min_samples'], algorithm='auto').fit(
            self.Xf[..., idx_signal][np.array([4, 6]), ...].T)
        s2 = time.time()

        print(f'Clustering : {np.round(s2 - s1)} s')
        # append cluster membership to windowed Xraw
        self.cl_labels = np.unique(dbs.labels_[dbs.labels_>-1], return_counts=True)

        cl_mem = np.ones(self.Xf.shape[1]) * -1
        cl_mem[idx_signal] = dbs.labels_
        self.Xf = np.r_[self.Xf, cl_mem[np.newaxis, ...]]  # Xf has 8 columns: scanIdOri, mz, int, st, scanIdNorm, noiseBool, st_adj, clMem
        # s3 = time.time()
        self.feat_summary1()
        s4 = time.time()

        print(f'total time pp (min): {np.round((s4-s0)/60, 1)}')

    def feat_summary1(self):
        # t1 = time.time()
        self.feat = {}
        self.feat_l2 = []
        self.feat_l3 = []
        print(f'Number of L0 features: {len(self.cl_labels[0])}')
        # filter for feature length
        min_le_bool = (self.cl_labels[1] >= self.qc_par['st_len_min']) & (self.cl_labels[0] != -1)
        cl_n_above = self.cl_labels[0][min_le_bool]
        print(f'Number of L1 features: {len(cl_n_above)} ({np.round(len(cl_n_above)/ len(self.cl_labels[0]) *100,1) }%)')
        self.feat.update(
            {'id:' + str(x): {'flev': 1, 'crit': 'n < st_len_min'} for x in self.cl_labels[0][~min_le_bool]})
        t2 = time.time()
        c = 0
        for fid in cl_n_above:
            c += 1
            # print(fid)
            f1 = self.cls(cl_id=fid, Xf=self.Xf, qc_par=self.qc_par)
            # print('---')
            ufid = 'id:' + str(fid)
            self.feat.update({ufid: f1})
            # print(f1['flev'])
            if f1['flev'] == 2:
                self.feat_l2.append(ufid)
            if f1['flev'] == 3:
                self.feat_l3.append(ufid)
        t3 = time.time()

        # qq = [self.feat[x]['qc'][0] if isinstance(self.feat[x]['qc'], type(())) else self.feat[x]['qc'] for x in
        #       self.feat_l2]

        print(f'Time for {len(cl_n_above)} feature summaries: {np.round(t3-t2)}s ({np.round(len(cl_n_above) / (t3-t2), 2)}s per ft)')

        if (len(self.feat_l2) == 0) & (len(self.feat_l3) == 0):
            raise ValueError('No L2 features found')
        else:
            print(f'Number of L2 features: {len(self.feat_l2)+len(self.feat_l3)} ({np.round((len(self.feat_l2)+len(self.feat_l3))/len(cl_n_above)*100)}%)')

        if len(self.feat_l3) == 0:
            # raise ValueError('No L3 features found')
            print('No L3 features found')
        else:
            print(f'Number of L3 features: {len(self.feat_l3)} ({np.round(len(self.feat_l3) / (len(self.feat_l2)+len(self.feat_l3))*100,1)}%)')

    # @background
    def cl_summary(self, cl_id: Union[int, np.int64]):
        # cl_id = cl_n_above[0]
        # cluster summary
        # X is data matrix
        # dbs is clustering object with lables
        # cl_id is cluster id in labels
        # st_adjust is scantime adjustment factor

        # Xf has 8 columns:
        # 0: scanIdOri,
        # 1: mz,
        # 2: Int,
        # 3: st,
        # 4: scanIdNorm,
        # 5: noiseBool,
        # 6: st_adj,
        # 7: clMem

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
        idx = np.where(self.Xf[7] == cl_id)[0]
        iLen = len(idx)

        # gaps or doublets in successive scan ids
        x_sid = self.Xf[4, idx]
        sid_diff = np.diff(x_sid)
        sid_gap = np.sum((sid_diff > 1) | (sid_diff < 1)) / (iLen - 1)

        # import matplotlib.pyplot as plt
        x = self.Xf[3, idx]  # scantime
        y = self.Xf[2, idx]  # intensity
        mz = self.Xf[1, idx]  # mz
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

        # _, icent = wImean(x, y)

        # minor smoothing - minimum three data points - and  padding
        ysm = np.convolve(y, np.ones(3) / 3, mode='valid')
        ysm = np.concatenate(([y[0]], ysm, [y[-1]]))
        xsm = x


        # I min-max scaling and bline correction
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
        scan_id = np.concatenate([np.where((self.Xf[4] == x) & (self.Xf[5] == 1))[0] for x in scans])
        noiI = np.median(self.Xf[2, scan_id])
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
        # mz_maxI = np.max(self.Xf[2, idx])
        mz_maxI = mz[np.argmax(y)]
        rt_maxI = x[np.argmax(y)]

        # rt_maxI = x[np.argmax(self.Xf[2, idx])]
        rt_mean = np.mean(xsm)
        mz_mean = np.mean(mz)

        # create subset with equal endings
        _, icent = wImean(xsm, ybcor)

        # tailing
        tail = rt_maxI / rt_mean

        # integrals
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

    # @background
    def cls(self, cl_id, Xf, qc_par):
        # cl_id = cl_n_above[0]
        # cluster summary
        # X is data matrix
        # dbs is clustering object with lables
        # cl_id is cluster id in labels
        # st_adjust is scantime adjustment factor

        # Xf has 8 columns:
        # 0: scanIdOri,
        # 1: mz,
        # 2: Int,
        # 3: st,
        # 4: scanIdNorm,
        # 5: noiseBool,
        # 6: st_adj,
        # 7: clMem

        qc = {}
        quant = {}
        descr = {}
        fdata = {}

        # run peak detection on smoothed version
        # split long signals int fwhm and 2*se

        # dp idx for cluster
        idx = np.where(Xf[7] == cl_id)[0]
        iLen = len(idx)
        descr['ndp'] = iLen

        # import matplotlib.pyplot as plt
        st = self.Xf[3, idx]  # scantime
        intens = self.Xf[2, idx]  # intensity
        mz = self.Xf[1, idx]  # mz
        fdata.update({'st': st, 'I_raw': intens, 'mz': mz})

        idx_maxI = np.argmax(intens)
        qc['icent'] = idx_maxI

        mz_maxI = mz[idx_maxI]
        mz_min = np.min(mz)
        mz_max = np.max(mz)

        rt_maxI = st[idx_maxI]
        rt_min = np.min(st)
        rt_max = np.max(st)

        st_span = rt_max - rt_min

        ppm = (mz_max - mz_min) / mz_maxI * 1e6
        qc['ppm'] = ppm
        descr.update({'mzMaxI': mz_maxI, 'rtMaxI': rt_maxI, 'st_span': st_span, 'mz_min': mz_min, 'mz_max': mz_max, 'rt_min': rt_min, 'rt_max': rt_max, })

        if (idx_maxI < 2) | (idx_maxI > (len(idx) - 2)):
            crit = 'icent: Peak is not centered'
            od = {'flev': 2, 'crit': crit, 'qc': qc, 'quant': quant, 'descr': descr, 'fdata': fdata}
            return od

        if ppm > qc_par['ppm']:
            crit = 'm/z variability of a feature (ppm)'
            od = {'flev': 2, 'crit': crit, 'qc': qc, 'quant': quant, 'descr': descr, 'fdata': fdata}
            return od

        # gaps or doublets in successive scan ids
        sid = Xf[4, idx]
        sid_diff = np.diff(sid)
        sid_gap = np.sum((sid_diff > 1) | (sid_diff < 1) | (sid_diff == 0)) / (iLen - 1)
        qc['sId_gap'] = sid_gap

        if sid_gap > qc_par['sId_gap']:
            crit = 'sId_gap: scan ids not consecutive (%)'
            od = {'flev': 2, 'crit': crit, 'qc': qc, 'quant': quant, 'descr': descr, 'fdata': fdata}
            return od


        # minor smoothing - minimum three data points - and  padding
        ysm = np.convolve(intens, np.ones(3) / 3, mode='valid')
        ysm = np.concatenate(([intens[0]], ysm, [intens[-1]]))
        xsm = st


        # I min-max scaling and bline correction
        intens_max = np.max(intens)
        intens = intens / intens_max
        ysm_max = np.max(ysm)
        ysm = ysm / ysm_max
        # bline correction of smoothed signal
        # if smoothed signal has less than three data points, then this won't work
        # minimum last points minu
        y_bl = ((np.min(ysm[-3:]) - np.min(ysm[0:3])) / (xsm[-1] - xsm[0])) * (xsm - np.min(xsm)) + np.min(ysm[0:3])
        ybcor = ysm - y_bl
        bcor_max = np.max(ybcor * ysm_max)

        fdata.update({'I_sm_bline': ybcor * ysm_max, 'I_smooth': ysm * ysm_max, 'mz': mz})
        nneg = np.sum(ybcor >= -0.05) / len(ysm)
        qc['non_neg'] = nneg
        if nneg < qc_par['non_neg']:
            crit = 'nneg: Neg Intensity values'
            od = {'flev': 2, 'crit': crit, 'qc': qc, 'quant': quant, 'descr': descr, 'fdata': fdata}
            return od
        # signal / noise
        scans = np.unique(sid)
        scan_id = np.concatenate([np.where((Xf[4] == x) & (Xf[5] == 1))[0] for x in scans])
        noiI = np.median(Xf[2, scan_id])
        sino = bcor_max / noiI
        qc['sino'] = sino
        if sino < qc_par['sino']:
            crit = 'sino: s/n below qc threshold'
            od = {'flev': 2, 'crit': crit, 'qc': qc, 'quant': quant, 'descr': descr, 'fdata': fdata}
            return od
        # intensity variation level
        idx_maxI = np.argmax(ybcor)
        sdxdy = (np.sum(np.diff(ybcor[:idx_maxI]) < (-0.05)) + np.sum(np.diff(ybcor[(idx_maxI + 1):]) > 0.05)) / (
                len(ybcor) - 3)
        qc['raggedness'] = sdxdy

        if sdxdy > qc_par['raggedness']:
            crit = 'raggedness: intensity variation level above qc threshold'
            od = {'flev': 2, 'crit': crit, 'qc': qc, 'quant': quant, 'descr': descr, 'fdata': fdata}
        # integrals
        #a = integrate.trapz(ybcor * bcor_max, x=xsm)
        a_raw = integrate.trapz(fdata['I_raw'], fdata['st'])
        a_smbl = integrate.trapz(fdata['I_sm_bline'], fdata['st'])
        quant.update({'raw': a_raw, 'smbl': a_smbl})
        # include symmetry and monotonicity
        pl = find_peaks(fdata['I_sm_bline'], distance=10)
        plle = len(pl[0])

        if plle > 0:
            pp = peak_prominences(fdata['I_sm_bline'], pl[0])
            if len(pp[0]) > 0:
                aa = str(np.round(pp[0][0]))
            else:
                aa = 0
        else:
            # pl = -1
            aa = -1
        descr.update({'npeaks': plle, 'pprom': aa,})

        od = {'flev': 3, 'qc': qc, 'quant': quant, 'descr': descr, 'fdata': fdata}
        return od

    @log
    def vizpp(self, selection: dict = {'mz_min': 425, 'mz_max': 440, 'rt_min': 400, 'rt_max': 440}, lev: int = 3):
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
        Xsub = self.window_mz_rt(self.Xf, selection, allow_none=False, return_idc=False)
        fid_window = np.unique(Xsub[7])
        l2_ffeat = list(compress(self.feat_l2, np.in1d([float(i.split(':')[1]) for i in self.feat_l2], fid_window)))
        l3_ffeat = list(compress(self.feat_l3, np.in1d([float(i.split(':')[1]) for i in self.feat_l3], fid_window)))

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
        psize = (imax/np.max(imax)) * 5
        im = axs[1].scatter(Xsub[3, idc_above], Xsub[1, idc_above], c=Xsub[2, idc_above], s=psize, cmap=cm, norm=LogNorm())

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
            l2l3_feat = l2_ffeat+l3_ffeat
            for i in l2l3_feat:
                print(i)
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
            vis_feature(fdict=self.feat[ii], id=ii, ax=axs[0])

        if lev == 3:
            for i in l3_ffeat:
                print(i)
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
            vis_feature(fdict=self.feat[ii], id=ii, ax=axs[0])
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
    def l3toDf(self):
        if len(self.feat_l3) > 0:
            descr = pd.DataFrame([self.feat[x]['descr'] for x in self.feat_l3])
            descr['id'] = self.feat_l3
            quant = pd.DataFrame([self.feat[x]['quant'] for x in self.feat_l3])
            qc = pd.DataFrame([self.feat[x]['qc'] for x in self.feat_l3])

            out =  pd.concat((descr[[ 'rt_maxI', 'mz_maxI', 'rt_min', 'rt_max', 'npeaks']], qc[['ppm', 'sino']], quant['smbl']), axis=1)
            out.index = descr['id']
            self.l3Df = out.sort_values(['rt_maxI', 'mz_maxI', 'smbl'], ascending=True)
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


class MSexp(Fdet, MSstat):
    @log
    def __init__(self, dpath: str, convert:bool = True, docker:dict={'repo': 'convo:v1', 'mslevel': 0, 'smode': [0, 1, 2, 3, 4, 5], 'amode': 2 , 'seg': [0, 1, 2, 3],} ): #  mslev='1', print_summary=False,
        self.mslevel = str(docker['mslevel']) if not isinstance(docker['mslevel'], list) else tuple(map(str, np.unique(docker['mslevel'])))
        if len(self.mslevel) == 1:
            self.mslevel = self.mslevel[0]
        self.smode = str(docker['smode']) if not isinstance(docker['smode'], list) else tuple(map(str, np.unique(docker['smode']).tolist()))
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
        # self.ecdf()
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
        if any((self.df.ScanMode == 2) & ~(self.df.MsLevel == 1)): # dda
            raise ValueError('Check AcquisitionKeys for scan types - combo not encountered before')

        if any((self.df.ScanMode == 5) & ~(self.df.MsLevel == 0)): # dia (bbcid)
            raise ValueError('Check AcquisitionKeys for scan types - combo not encountered before')
        idx_dda = (self.df.ScanMode == 2) & (self.df.ScanMode == 1)
        if any(idx_dda):
            print('DDA')
        idx_dia = (self.df.ScanMode == 5) & (self.df.MsLevel == 0)
        if any(idx_dia):
            print('DIA')
        if any(idx_dia) & any(idx_dda):
            raise ValueError('Check AcquisitionKeys for scan types - combo dda and dia not encountered before')
        idx_ms0 = (self.df.ScanMode == 0) & (self.df.MsLevel == 0)

        self.ms0string = self.df['stype'][idx_ms0].unique()[1:][0]
        if any(~idx_ms0):
            self.ms1string = self.df['stype'][~idx_ms0].unique()[0]
        else:
            self.ms1string: None
        self.df['LevPP'] = None
        add =  self.df['LevPP'].copy()
        add.loc[idx_dda] = 'dda'
        add.loc[idx_dia] = 'dia'
        add.loc[idx_ms0] = 'fs'
        self.df['LevPP'] = add

    def rawd(self, ss):
        # create dict for each scantype
        xr = {i: {'Xraw': [], 'df': []} for i in self.df.stype.unique()}
        c = {i: 0 for i in self.df.stype.unique()}
        for i in range(self.df.shape[0]):
            of = np.ones_like(ss['LineIndexId'][i][1])
            sid = of * i
            mz = ss['LineMzId'][i][1]
            intens = ss['LineIntensityId'][i][1]
            st = of * ss['sinfo'][i]['Rt']
            sid1 = np.ones_like(ss['LineIndexId'][i][1]) * c[self.df.stype.iloc[i]]
            xr[self.df.stype.iloc[i]]['Xraw'].append(np.array([sid, mz, intens, st, sid1]))
            c[self.df.stype.iloc[i]] += 1

        self.xrawd = {i: np.concatenate(xr[i]['Xraw'], axis=1) for i in self.df.stype.unique()}
        self.dfd = {i: self.df[self.df.stype == i] for i in self.df.stype.unique()}

    @log
    def read_mm8(self):
        ss = pickle.load(open(self.msmfile, 'rb'))
        self.df = pd.DataFrame(ss['sinfo'])
        # define ms level for viz and peak picking
        idcS3 = (self.df.Segment == 3).values
        if any(idcS3):
            self.df['stype'] = "0"
            add = self.df['stype'].copy()
            add[idcS3] = [f'{x["MsLevel"]}_{x["AcquisitionKey"]}_{x["AcquisitionMode"]}_{x["ScanMode"]}_{x["Collision_Energy_Act"]}' for x in ss['sinfo'] if x['Segment'] == 3]
            self.df['stype'] = add

        self.msZeroPP()

        # create dict for each scantype and df
        self.rawd(ss)

        polmap = {0: 'P', 1: 'N'}
        df1 = pd.DataFrame(
            [(polmap[x['Polarity']], str(x['MzAcqRangeLower']) + '-' + str(x['MzAcqRangeUpper']), x['MsLevel']) for x in ss['sinfo']], columns=['Polarity', 'MzAcqRange', 'MsLevel'])
        df1 = pd.DataFrame([self.csummary(i, df1) for i in range(df1.shape[1])], index=df1.columns).transpose()
        df1['nSc'] = len(ss['sinfo'])
        df2 = pd.DataFrame([(x['AcquisitionMode'], x['ScanMode'], x['Segment']) for x in ss['sinfo']],
                           columns=['AcqMode', 'ScMode', 'Segments'])
        t = pd.DataFrame(df2.groupby(['Segments']).apply(lambda x: self.vc(x, n=df2.shape[0])))
        t = t.rename(index={t.index[i]: 'Segment ' + str(t.index[i]) for i in range(t.shape[0])}).transpose()
        self.summary = pd.concat([df1, t], axis=1)

        # asps = 1 / ((self.df.Rt.max() - self.df.Rt.min()) /  df1['nSc'].iloc[0])  # hz
        # ii = np.diff(np.unique(self.df.Rt))
        # sps = pd.DataFrame(
        #     {'ScansPerSecAver': asps, 'ScansPerSecMin': 1 / ii.max(), 'ScansPerSecMax': 1 / ii.min()}, index=[0])
        # self.summary = pd.concat([df1, t, sps], axis=1)
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
        part_none = any([x == None for x in selection.values()])
        all_fill = any([x != None for x in selection.values()])
        if not allow_none:
            if not all_fill:
                raise ValueError('Define plot boundaries with selection argument.')
        if part_none:
            if selection['mz_min'] is not None:
                t1 = Xr[1] >= selection['mz_min']
            else:
                t1 = np.ones(Xr.shape[1], dtype=bool)
            if selection['mz_max'] is not None:
                t2 = Xr[1] <= selection['mz_max']
            else:
                t2 = np.ones(Xr.shape[1], dtype=bool)
            if selection['rt_min'] is not None:
                t3 = Xr[3] > selection['rt_min']
            else:
                t3 = np.ones(Xr.shape[1], dtype=bool)
            if selection['rt_max'] is not None:
                t4 = Xr[3] < selection['rt_max']
            else:
                t4 = np.ones(Xr.shape[1], dtype=bool)
        else:
            t1 = Xr[1] >= selection['mz_min']
            t2 = Xr[1] <= selection['mz_max']
            t3 = Xr[3] > selection['rt_min']
            t4 = Xr[3] < selection['rt_max']
        idx = np.where(t1 & t2 & t3 & t4)[0]
        if return_idc:
            return idx
        else:
            return Xr[..., idx]

    @staticmethod
    def tick_conv(X):
        V = X / 60
        return ["%.2f" % z for z in V]

    def _noiseT(self, p, X=None, local=True):
        if local:
            Xs = X
        else:
            Xs = self.xrawd[self.ms0string]
        noise_thres = np.quantile(Xs[2], q=p)

        idxN = Xs[2] > noise_thres
        idc_above = np.where(idxN)[0]
        idc_below = np.where(~idxN)[0]
        return (idc_below, idc_above)

    # @log
    @staticmethod
    def get_density(X:np.ndarray, bw:float, q_noise:float=0.5):
        xmin = X[1].min()
        xmax = X[1].max()
        b = np.linspace(xmin, xmax, 500)
        idxf = np.where(X[2] > np.quantile(X[2], q_noise))[0]
        x_rev = X[1, idxf]
        kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(x_rev[:, np.newaxis])
        log_dens = np.exp(kde.score_samples(b[:, np.newaxis]))
        return (b, log_dens / np.max(log_dens))

    @log
    def viz1(self, q_noise:float=0.50, selection:dict={'mz_min': None, 'mz_max': None, 'rt_min': None, 'rt_max': None}, qcm_local: bool=True):
        @log
        def on_lims_change(event_ax: plt.axis):
            x1, x2 = event_ax.get_xlim()
            y1, y2 = event_ax.get_ylim()
            selection = {'mz_min': y1, 'mz_max': y2, 'rt_min': x1, 'rt_max': x2}
            Xs = self.window_mz_rt(self.xrawd[self.ms0string], selection, allow_none=False)
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
        Xsub = self.window_mz_rt(self.xrawd[self.ms0string], selection, allow_none=False)
        idc_below, idc_above = self._noiseT(p=q_noise, X=Xsub, local=qcm_local)
        cm = plt.cm.get_cmap('rainbow')
        fig, axs = plt.subplots(1, 2, sharey=False, gridspec_kw={'width_ratios': [3, 1]})
        axs[1].text(1.05, 0, self.fname, rotation=90, fontsize=4, transform=axs[1].transAxes)
        axs[1].tick_params(axis='y', direction='in')
        axs[0].scatter(Xsub[3, idc_below], Xsub[1, idc_below], s=0.1, c='gray', alpha=0.5)
        im = axs[0].scatter(Xsub[3, idc_above], Xsub[1, idc_above], c=Xsub[2, idc_above], s=5, cmap=cm,
                            norm=LogNorm())
        cbaxes = fig.add_axes([0.15, 0.77, 0.021, 0.1])
        cb = fig.colorbar(mappable=im, cax=cbaxes, orientation='vertical',
                          format=LogFormatterSciNotation(base=10, labelOnlyBase=False))
        cb.ax.tick_params(labelsize='xx-small')
        fig.subplots_adjust(wspace=0.05)
        axs[0].callbacks.connect('ylim_changed', on_lims_change)
        return (fig, axs)

    @log
    def viz(self, q_noise: float = 0.89, selection: dict = {'mz_min': None, 'mz_max': None, 'rt_min': None, 'rt_max': None}, qcm_local: bool=True):
        Xsub = self.window_mz_rt(self.xrawd[self.ms0string], selection, allow_none=False)
        idc_below, idc_above = self._noiseT(p=q_noise, X=Xsub, local=qcm_local)
        cm = plt.cm.get_cmap('rainbow')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.text(1.25, 0, self.dpath, rotation=90, fontsize=6, transform=ax.transAxes)
        ax.scatter(Xsub[3, idc_below], Xsub[1, idc_below], s=0.1, c='gray', alpha=0.5)
        im = ax.scatter(Xsub[3, idc_above], Xsub[1, idc_above], c=Xsub[2, idc_above], s=5, cmap=cm, norm=LogNorm())
        fig.canvas.draw()
        ax.set_xlabel(r"$\bfScan time$ [sec]")
        ax2 = ax.twiny()
        ax.yaxis.offsetText.set_visible(False)
        ax.yaxis.set_label_text(r"$\bfm/z$")
        tmax = np.max(Xsub[3])
        tmin = np.min(Xsub[3])
        tick_loc = np.linspace(tmin / 60, tmax / 60, 5) * 60
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(tick_loc)
        ax2.set_xticklabels(self.tick_conv(tick_loc))
        ax2.set_xlabel(r"[min]")
        fig.colorbar(im, ax=ax)
        fig.show()
        return (fig, ax)

    def massSpectrumDIA(self, rt:float=150., scantype:str='0_2_2_5_20.0', viz:bool=False):
        import matplotlib.pyplot as plt
        import numpy as np
        df = self.dfd[scantype]
        idx = np.argmin(np.abs(df.Rt-rt))
        inf = df.iloc[idx]
        sub = self.xrawd[scantype]
        sub = sub[..., sub[0] == (inf.Id-1)]
        if viz:
            f, ax = plt.subplots(1, 1)
            mz = sub[1]
            intens =  sub[2]
            ax.vlines(sub[1], np.zeros_like(mz), intens/np.max(intens) * 100)
            ax.text(0.77, 0.95, f'Acq Type: {inf.LevPP.upper()}', rotation=0, fontsize=8, transform=ax.transAxes, ha='left')
            ax.text(0.77, 0.9, f'Rt: {inf.Rt} s', rotation=0, fontsize=8, transform=ax.transAxes, ha='left')
            ax.text(0.77, 0.85, f'Collision E: {inf.Collision_Energy_Act} eV', rotation=0, fontsize=8, transform=ax.transAxes, ha='left')
            return (f, ax)
        else:
            return (sub[1], sub[2], np.zeros_like(sub[0]), inf)

    def ms2L(self, q_noise, selection, qcm_local: bool = True):
        mpl.rcParams['keymap.back'].remove('left') if ('left' in mpl.rcParams['keymap.back']) else None
        # def fwd(x):
        #     return (x / self.intensmax) * 100
        # def bwd(x):
        #     return (x / 100) * self.intensmax
        def on_lims_change(event_ax: plt.axis):
            x1, x2 = event_ax.get_xlim()
            y1, y2 = event_ax.get_ylim()
            selection = {'mz_min': y1, 'mz_max': y2, 'rt_min': x1, 'rt_max': x2}
            Xs = self.window_mz_rt(self.xrawd[self.ms0string], selection, allow_none=False)
            bsMin = (y2 - y1) / 200 if ((y2 - y1) / 200) > 0.0001 else 0.0001
            bw = bsMin
            y, x = self.get_density(Xs, bw, q_noise=q_noise)
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
                axs[0,0].clear()
                mz, intens, ylim0, inf = self.massSpectrumDIA(rt=self.rtPlot, scantype=self.ms1string, viz=False)

            if event_ax.key == 'right':
                self.rtPlot = self.dfd[self.ms1string].iloc[np.argmin(np.abs(self.rtPlot - self.dfd[self.ms1string].Rt))+1].Rt
                xl = axs[0, 0].get_xlim()
                axs[0, 0].clear()
                mz, intens, ylim0, inf = self.massSpectrumDIA(rt=self.rtPlot, scantype=self.ms1string, viz=False)
                axs[0, 0].set_xlim(xl)

            if event_ax.key == 'left':
                self.rtPlot = self.dfd[self.ms1string].iloc[np.argmin(np.abs(self.rtPlot - self.dfd[self.ms1string].Rt)) - 1].Rt
                xl = axs[0, 0].get_xlim()
                axs[0, 0].clear()
                mz, intens, ylim0, inf = self.massSpectrumDIA(rt=self.rtPlot, scantype=self.ms1string, viz=False)
                axs[0, 0].set_xlim(xl)

            axs[0, 0].vlines(mz, ylim0, intens)
            axs[0, 0].text(0.77, 0.85, f'Acq Type: {inf.LevPP.uppder() if inf.LevPP is not None else "-"}', rotation=0, fontsize='xx-small',
                           transform=axs[0, 0].transAxes, ha='left')
            axs[0, 0].text(0.77, 0.75, f'Rt: {np.round(inf.Rt, 2)} s ({inf.Id})', rotation=0, fontsize='xx-small',
                           transform=axs[0, 0].transAxes, ha='left')
            axs[0, 0].text(0.77, 0.65, f'Collision E: {inf.Collision_Energy_Act} eV', rotation=0, fontsize='xx-small',
                           transform=axs[0, 0].transAxes, ha='left')
            axs[0, 0].text(0.77, 0.55, f'Parent SID: {inf.Parent}', rotation=0, fontsize='xx-small',
                           transform=axs[0, 0].transAxes, ha='left')
            axs[0, 0].text(0.77, 0.45, f'Isol mass: {np.round(inf.MSMS_IsolationMass_Act, 2)}', rotation=0, fontsize='xx-small',
                           transform=axs[0, 0].transAxes, ha='left')
            axs[0, 0].text(0.77, 0.35, f'Q Iso-Res: {np.round(inf.Quadrupole_IsolationResolution_Act, 2)}', rotation=0,
                           fontsize='xx-small', transform=axs[0, 0].transAxes, ha='left')


            d = (axs[1, 0].transData + axs[1, 0].transAxes.inverted()).transform((self.rtPlot, 100))
            self.xind = axs[1, 0].scatter([d[0]], [-0.01], c='red', clip_on=False, marker='^', transform=axs[1, 0].transAxes)
            #self.intensmax = np.max(intens)
            #secax = axs[0, 0].secondary_yaxis('right', functions=(fwd, bwd))


        Xsub = self.window_mz_rt(self.xrawd[self.ms0string], selection, allow_none=False)
        idc_below, idc_above = self._noiseT(p=q_noise, X=Xsub, local=qcm_local)

        cm = plt.cm.get_cmap('rainbow')
        fig, axs = plt.subplots(2, 2, sharey='row', gridspec_kw={'width_ratios': [3, 1], 'height_ratios': [1.5, 3]})
        axs[1, 1].text(1.05, 0, self.dpath, rotation=90, fontsize=4, transform=axs[1, 1].transAxes)
        axs[1, 1].tick_params(axis='y', direction='in')
        axs[1, 0].scatter(Xsub[3, idc_below], Xsub[1, idc_below], s=0.1, c='gray', alpha=0.5)
        im = axs[1, 0].scatter(Xsub[3, idc_above], Xsub[1, idc_above], c=(Xsub[2, idc_above]), s=5, cmap=cm, norm=LogNorm())
        cbaxes = fig.add_axes([0.15, 0.44, 0.021, 0.1])
        cb = fig.colorbar(mappable=im, cax=cbaxes, orientation='vertical',
                          format=LogFormatterSciNotation(base=10, labelOnlyBase=False))
        cb.ax.tick_params(labelsize='xx-small')
        fig.subplots_adjust(wspace=0.05)
        axs[1, 0].callbacks.connect('ylim_changed', on_lims_change)
        axs[1, 0].callbacks.connect('ylim_changed', on_lims_change)
        cid = fig.canvas.mpl_connect('key_press_event', on_key)

        self.rtPlot = Xsub[3, np.argmax(Xsub[2])]
        mz, intens, ylim0, inf = self.massSpectrumDIA(rt=self.rtPlot, scantype = self.ms1string, viz= False)
        #self.intensmax = np.max(intens)
        axs[0,0].vlines(mz, ylim0, intens)
        axs[0,0].text(0.75, 0.85, f'Acq Type: {inf.LevPP.uppder() if inf.LevPP is not None else "-"}', rotation=0, fontsize='xx-small', transform=axs[0,0].transAxes, ha='left')
        axs[0,0].text(0.75, 0.75, f'Rt: {inf.Rt} s ({inf.Id})', rotation=0, fontsize='xx-small', transform=axs[0,0].transAxes, ha='left')
        axs[0,0].text(0.75, 0.65, f'Collision E: {inf.Collision_Energy_Act} eV', rotation=0, fontsize='xx-small', transform=axs[0,0].transAxes, ha='left')
        axs[0,0].text(0.75, 0.55, f'Parent SID: {inf.Parent}', rotation=0, fontsize='xx-small', transform=axs[0,0].transAxes, ha='left')
        axs[0, 0].text(0.75, 0.45, f'Isol mass: {np.round(inf.MSMS_IsolationMass_Act, 2)}', rotation=0, fontsize='xx-small', transform=axs[0, 0].transAxes, ha='left')
        axs[0, 0].text(0.75, 0.35, f'Q Iso-Res: {np.round(inf.Quadrupole_IsolationResolution_Act, 2)}', rotation=0, fontsize='xx-small', transform=axs[0, 0].transAxes, ha='left')
        axs[0, 0].tick_params(axis='y', direction='out')
        axs[0,1].set_axis_off()

        axs[0,0].xaxis.set_label_text(r"$\bfm/z$")
        axs[1,0].set_xlabel(r"$\bfScan time$ [sec]")
        axs[1,0].yaxis.set_label_text(r"$\bfm/z$")

        d = (axs[1, 0].transData + axs[1, 0].transAxes.inverted()).transform((self.rtPlot, 100))
        self.xind = axs[1, 0].scatter([d[0]], [-0.01], c='red', clip_on=False, marker='^', transform=axs[1, 0].transAxes)
        fig.subplots_adjust(wspace=0.05, hspace=0.3)

    @log
    def chromg(self, tmin:float=None, tmax:float=None, ctype:list =['tic', 'bpc', 'xic'], xic_mz:float =[], xic_ppm:float=10):
        df_l1 = self.dfd[self.ms0string]
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
        tick_loc = np.arange(np.round(tmin/60), np.round(tmax/60), 2, dtype=float) * 60
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xticks(tick_loc)
        ax2.set_xticklabels(self.tick_conv(tick_loc))
        ax2.set_xlabel(r"$\bfScantime (min)$")
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

        idx_mz = np.where((self.xrawd[self.ms0string][1] >= mz_min) & (self.xrawd[self.ms0string] <= mz_max))[0]
        if len(idx_mz) < 0:
            raise ValueError('mz range not found')

        X_mz = self.xrawd[self.ms0string][..., idx_mz]

        sid = np.array(np.unique(X_mz[0]).astype(int))
        xic = np.zeros(int(np.max(self.xrawd[self.ms0string][0])+1))
        for i in sid:
            xic[i-1] = np.sum(X_mz[np.where(X_mz[:,0] == i), 2])
        stime = np.sort(np.unique(self.xrawd[self.ms0string][3]))

        if (~isinstance(rt_min, type(None)) | ~isinstance(rt_max, type(None))):
            idx_rt = np.where((stime >= rt_min) & (stime <= rt_max))[0]
            if len(idx_rt) < 0:
                raise ValueError('rt range not found')
            stime = stime[idx_rt]
            xic = xic[idx_rt]

        return (stime, xic)

    @staticmethod
    def fwhmBound(intens, st, rtDelta=1-0.4):
        idxImax = np.argmax(intens)
        ifwhm = intens[idxImax] / 2

        if (idxImax == 0) or (idxImax == len(intens)):
            return None

        st_imax = st[idxImax]

        ll = np.max(st[(intens < ifwhm) & (st <= st_imax)])
        ul = np.min(st[(intens > ifwhm) & (st >= st_imax)])
        ll1 = st_imax - ((ul - ll) * rtDelta) / 2
        ul1 = st_imax + ((ul - ll) * rtDelta) / 2

        return (ul1, ll1)


    def igr(self):
        self.l3toDf()
        au = self.l3Df
        rtd = 0.6 # fwhm proportion
        fset = set(au.index.values)
        fgr = []
        c =0
        for i in fset:
            intens = self.feat[i]['fdata']['I_sm_bline']
            st=self.feat[i]['fdata']['st_sm']
            out = self.fwhmBound(intens, st, rtDelta=rtd)

            if out is None:
                continue

            sub = au[(au.rt_maxI < out[0]) & (au.rt_maxI > out[1])]
            if sub.shape[0] <= 1:
                continue

            sidx = sub.index[np.argmax(sub.smbl.values)]
            intens1 = self.feat[sidx]['fdata']['I_sm_bline']
            st1 = self.feat[sidx]['fdata']['st_sm']
            out1 = self.fwhmBound(intens1, st1, rtDelta=rtd)
            sub1 = au[(au.rt_maxI < out1[0]) & (au.rt_maxI > out1[1])].copy()
            sub1['fgr']=c
            sub1 = sub1.sort_values('mz_maxI')
            c+=1

            if any(sub1['smbl'] > 1e6):
                fgr.append(sub1)
            fset = fset - set(sub1.index.values)

        return pd.concat(fgr) # candidate isotopologues