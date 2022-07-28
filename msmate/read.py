# import pandas as pd
# import numpy as np
# import re
# import xml.etree.cElementTree as ET
# import obonet as obo
# import os
# import docker as dock
# import time
# import pickle
# import base64
# import zlib
#
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# from matplotlib.backend_bases import MouseButton
# from matplotlib.patches import Rectangle
# from matplotlib.widgets import RectangleSelector
# from matplotlib.colors import LogNorm
# from matplotlib.ticker import LogFormatterSciNotation
#
# from itertools import compress
#
# from sklearn.cluster import DBSCAN
# from sklearn.neighbors import KernelDensity
# from scipy.signal import find_peaks, peak_prominences
# from scipy import integrate
#
# # combine mzml and docker d file read in into single class using class methods
# from msmate.impMzf import collect_spectra_chrom, get_obo, children, node_attr_recurse
# from msmate.runP import MSstat, Fdet
#
# class ReadB:
#     def __init__(self, dpath: str, convert: bool = True,
#                  docker: dict = {'repo': 'convo:v1', 'mslevel': 0, 'smode': [0, 1, 2, 3, 4, 5], 'amode': 2,
#                                  'seg': [0, 1, 2, 3], }):
#         self.mslevel = str(docker['mslevel']) if not isinstance(docker['mslevel'], list) else tuple(
#             map(str, np.unique(docker['mslevel'])))
#         if len(self.mslevel) == 1:
#             self.mslevel = self.mslevel[0]
#         self.smode = str(docker['smode']) if not isinstance(docker['smode'], list) else tuple(
#             map(str, np.unique(docker['smode']).tolist()))
#         if len(self.smode) == 1:
#             self.smode = self.smode[0]
#         self.amode = str(docker['amode'])
#         self.seg = str(docker['seg']) if not isinstance(docker['seg'], list) else tuple(
#             map(str, np.unique(docker['seg'])))
#         if len(self.seg) == 1:
#             self.seg = self.seg[0]
#         self.docker = {'repo': docker['repo'], 'mslevel': self.mslevel, 'smode': self.smode, 'amode': self.amode,
#                        'seg': self.seg, }
#         self.dpath = os.path.abspath(dpath)
#         self.dbfile = os.path.join(dpath, 'analysis.sqlite')
#         self.fname = os.path.basename(dpath)
#         self.msmfile = os.path.join(dpath,
#                                     f'mm8v3_edata_msl{self.docker["mslevel"]}_sm{"".join(self.docker["smode"])}_am{self.docker["amode"]}_seg{"".join(self.docker["seg"])}.p')
#         if os.path.exists(self.msmfile):
#             self.read_mm8()
#         else:
#             if convert:
#                 self.convoDocker()
#                 self.read_mm8()
#             else:
#                 raise SystemError('d folder requries conversion, enable docker convo')
#         # self.ecdf()
#
#     def convoDocker(self):
#         t0 = time.time()
#         client = dock.from_env()
#         client.info()
#         client.containers.list()
#         # img = [x for x in client.images.list() if self.docker["repo"] in x.tags]
#         img = [x for x in client.images.list() if self.docker["repo"] in x.tags]
#         if len(img) < 1:
#             raise ValueError('Image not found')
#         ec = f'docker run -v "{os.path.dirname(self.dpath)}":/data {self.docker["repo"]} "edata" "{self.fname}" -l {self.docker["mslevel"]} -am {self.docker["amode"]} -sm {" ".join(self.docker["smode"])} -s {" ".join(self.docker["seg"])}'
#         print(ec)
#         os.system(ec)
#         t1 = time.time()
#         print(f'Conversion time: {np.round(t1 - t0)} sec')
#
#     def msZeroPP(self):
#         if any((self.df.ScanMode == 2) & ~(self.df.MsLevel == 1)):  # dda
#             raise ValueError('Check AcquisitionKeys for scan types - combo not encountered before')
#
#         if any((self.df.ScanMode == 5) & ~(self.df.MsLevel == 0)):  # dia (bbcid)
#             raise ValueError('Check AcquisitionKeys for scan types - combo not encountered before')
#         idx_dda = (self.df.ScanMode == 2) & (self.df.ScanMode == 1)
#         if any(idx_dda):
#             print('DDA')
#         idx_dia = (self.df.ScanMode == 5) & (self.df.MsLevel == 0)
#         if any(idx_dia):
#             print('DIA')
#         if any(idx_dia) & any(idx_dda):
#             raise ValueError('Check AcquisitionKeys for scan types - combo dda and dia not encountered before')
#         idx_ms0 = (self.df.ScanMode == 0) & (self.df.MsLevel == 0)
#
#         self.ms0string = self.df['stype'][idx_ms0].unique()[1:][0]
#         if any(~idx_ms0):
#             self.ms1string = self.df['stype'][~idx_ms0].unique()[0]
#         else:
#             self.ms1string: None
#         self.df['LevPP'] = None
#         add = self.df['LevPP'].copy()
#         add.loc[idx_dda] = 'dda'
#         add.loc[idx_dia] = 'dia'
#         add.loc[idx_ms0] = 'fs'
#         self.df['LevPP'] = add
#
#     def rawd(self, ss):
#         # create dict for each scantype
#         xr = {i: {'Xraw': [], 'df': []} for i in self.df.stype.unique()}
#         c = {i: 0 for i in self.df.stype.unique()}
#         for i in range(self.df.shape[0]):
#             of = np.ones_like(ss['LineIndexId'][i][1])
#             sid = of * i
#             mz = ss['LineMzId'][i][1]
#             intens = ss['LineIntensityId'][i][1]
#             st = of * ss['sinfo'][i]['Rt']
#             sid1 = np.ones_like(ss['LineIndexId'][i][1]) * c[self.df.stype.iloc[i]]
#             xr[self.df.stype.iloc[i]]['Xraw'].append(np.array([sid, mz, intens, st, sid1]))
#             c[self.df.stype.iloc[i]] += 1
#
#         self.xrawd = {i: np.concatenate(xr[i]['Xraw'], axis=1) for i in self.df.stype.unique()}
#         self.dfd = {i: self.df[self.df.stype == i] for i in self.df.stype.unique()}
#
#     def read_mm8(self):
#         ss = pickle.load(open(self.msmfile, 'rb'))
#         self.df = pd.DataFrame(ss['sinfo'])
#         # define ms level for viz and peak picking
#         idcS3 = (self.df.Segment == 3).values
#         if any(idcS3):
#             self.df['stype'] = "0"
#             add = self.df['stype'].copy()
#             add[idcS3] = [
#                 f'{x["MsLevel"]}_{x["AcquisitionKey"]}_{x["AcquisitionMode"]}_{x["ScanMode"]}_{x["Collision_Energy_Act"]}'
#                 for x in ss['sinfo'] if x['Segment'] == 3]
#             self.df['stype'] = add
#
#         self.msZeroPP()
#
#         # create dict for each scantype and df
#         self.rawd(ss)
#
#         polmap = {0: 'P', 1: 'N'}
#         df1 = pd.DataFrame(
#             [(polmap[x['Polarity']], str(x['MzAcqRangeLower']) + '-' + str(x['MzAcqRangeUpper']), x['MsLevel']) for x in
#              ss['sinfo']], columns=['Polarity', 'MzAcqRange', 'MsLevel'])
#         df1 = pd.DataFrame([self.csummary(i, df1) for i in range(df1.shape[1])], index=df1.columns).transpose()
#         df1['nSc'] = len(ss['sinfo'])
#         df2 = pd.DataFrame([(x['AcquisitionMode'], x['ScanMode'], x['Segment']) for x in ss['sinfo']],
#                            columns=['AcqMode', 'ScMode', 'Segments'])
#         t = pd.DataFrame(df2.groupby(['Segments']).apply(lambda x: self.vc(x, n=df2.shape[0])))
#         t = t.rename(index={t.index[i]: 'Segment ' + str(t.index[i]) for i in range(t.shape[0])}).transpose()
#         self.summary = pd.concat([df1, t], axis=1)
#
#
#     @staticmethod
#     def csummary(i: int, df: pd.DataFrame):
#         if df.iloc[:, i].isnull().all():
#             return None
#         s = np.unique(df.iloc[:, i], return_counts=True)
#         n = df.shape[0]
#         if len(s[0]) < 5:
#             sl = []
#             for i in range(len(s[0])):
#                 sl.append(f'{s[0][i]} ({np.round(s[1][i] / n * 100)}%)')
#             return '; '.join(sl)
#         else:
#             return f'{np.round(np.mean(s), 1)} ({np.round(np.min(s), 1)}-{np.round(np.max(s), 1)})'
#
#     @staticmethod
#     def vc(x, n):
#         import numpy as np
#         ct = x.value_counts(subset=['ScMode', 'AcqMode'])
#         s = []
#         for i in range(ct.shape[0]):
#             s.append(
#                 f'{ct.index.names[0]}={ct.index[i][0]} & {ct.index.names[1]}={ct.index[i][1]}: {ct[ct.index[i]]} ({np.round(ct[ct.index[0]] / n * 100, 1)} %)')
#         return '; '.join(s)
#
#     @staticmethod
#     def window_mz_rt(Xr: np.ndarray, selection: dict = {'mz_min': None, 'mz_max': None, 'rt_min': None, 'rt_max': None},
#                      allow_none: bool = False, return_idc: bool = False):
#         part_none = any([x == None for x in selection.values()])
#         all_fill = any([x != None for x in selection.values()])
#         if not allow_none:
#             if not all_fill:
#                 raise ValueError('Define plot boundaries with selection argument.')
#         if part_none:
#             if selection['mz_min'] is not None:
#                 t1 = Xr[1] >= selection['mz_min']
#             else:
#                 t1 = np.ones(Xr.shape[1], dtype=bool)
#             if selection['mz_max'] is not None:
#                 t2 = Xr[1] <= selection['mz_max']
#             else:
#                 t2 = np.ones(Xr.shape[1], dtype=bool)
#             if selection['rt_min'] is not None:
#                 t3 = Xr[3] > selection['rt_min']
#             else:
#                 t3 = np.ones(Xr.shape[1], dtype=bool)
#             if selection['rt_max'] is not None:
#                 t4 = Xr[3] < selection['rt_max']
#             else:
#                 t4 = np.ones(Xr.shape[1], dtype=bool)
#         else:
#             t1 = Xr[1] >= selection['mz_min']
#             t2 = Xr[1] <= selection['mz_max']
#             t3 = Xr[3] > selection['rt_min']
#             t4 = Xr[3] < selection['rt_max']
#         idx = np.where(t1 & t2 & t3 & t4)[0]
#         if return_idc:
#             return idx
#         else:
#             return Xr[..., idx]
#
# class ReadM:
#     def __init__(self, fpath: str, mslev:str='1', print_summary:bool=False):
#         self.fpath = fpath
#         if mslev != '1':
#             raise ValueError('Check mslev argument - This function currently support ms level 1 import only.')
#         self.mslevel = mslev
#         self.print_summary = print_summary
#
#         # single level read
#         self.read_mzml()
#         self.createSpectMat()
#         self.stime_conv()
#         self.ms0string = 'mzml_msl1'
#         self.ms1string = None
#
#         self.xrawd = {self.ms0string: self.Xraw}
#         self.dfd = {self.ms0string: self.df}
#
#     def read_mzml(self):
#         # this is for mzml version 1.1.0
#         # schema specification: https://raw.githubusercontent.com/HUPO-PSI/mzML/master/schema/schema_1.1/mzML1.1.0.xsd
#         # read in data, files index 0 (see below)
#         # flag is 1/2 for msLevel 1 or 2
#         tree = ET.parse(self.fpath)
#         root = tree.getroot()
#         child = children(root)
#         imzml = child.index('mzML')
#         mzml_children = children(root[imzml])
#         obos = root[imzml][mzml_children.index('cvList')] # controlled vocab (CV
#         self.obo_ids = get_obo(obos, obo_ids={})
#         seq = np.where(~np.isin(mzml_children, ['cvList', 'run']))[0]
#         pp = {}
#         for j in seq:
#             filed = node_attr_recurse(s=root[imzml][j], d=4, c=0, ii=[])
#             dn = {}
#             for i in range(len(filed)):
#                 dn.update(dict(zip([filed[i]['path'] + '_' + re.sub('\{.*\}', '', x) for x in list(filed[i].keys())[1:]],
#                                    list(filed[i].values())[1:])))
#             pp.update({mzml_children[j]: dn})
#         run = root[imzml][mzml_children.index('run')]
#         self.out31 = collect_spectra_chrom(s=run, ii={}, d=20, c=0, flag=self.mslevel, tag='', obos=self.obo_ids)
#
#     def createSpectMat(self):
#         sc_msl1 = [i for i, x in self.out31.items() if len(x['data']) == 2] # sid of ms level 1 scans
#         # summarise and collect data
#         ms = []
#         meta = []
#         sid_count = 0
#         for i in sc_msl1:
#             # if ('s' in i) : # and (d['meta']['MS:1000511'] == '1')   # scans, obo ms:1000511 -> ms level
#             d = self.out31[i]
#             meta.append(d['meta'])
#             add = []
#             for j, dd in (d['data']).items():
#                 add.append(dd['d'])
#             idm = np.ones_like(dd['d'])
#             add.append(idm * sid_count)
#             add.append(idm * float(d['meta']['MS:1000016'])) # obo ms:1000016 -> scan start time
#             ms.append((i, np.array(add)))
#             sid_count += 1
#
#         self.Xraw = np.concatenate([x[1] for x in ms], axis=1)
#         self.Xraw = self.Xraw[[3, 0, 1, 2, 3]] #[sid, mz, intens, st, sid1]
#
#
#         self.df = pd.DataFrame(meta, index=sc_msl1)
#
#         if 'MS:1000129' in self.df.columns:
#             self.df['MS:1000129'] = 'N (-)'
#         if 'MS:1000130' in self.df.columns:
#             self.df['MS:1000130'] = 'P (+)'
#         if 'MS:1000127' in self.df.columns:
#             add = np.repeat('centroided', self.df.shape[0])
#             add[~(self.df['MS:1000127'] == True)] = 'profile'
#             self.df['MS:1000127'] = add
#         if 'MS:1000128' in self.df.columns:
#             add = np.repeat('profile', self.df.shape[0])
#             add[~(self.df['MS:1000128'] == True)] = 'centroided'
#             self.df['MS:1000128'] = add
#
#         self.df = self.df.rename(
#             columns={'UO:0000010': 'time_unit', 'UO:0000031': 'time_unit', 'defaultArrayLength': 'n',
#                      'MS:1000129': 'scanPolarity', 'MS:1000130': 'scanPolarity',
#                      'MS:1000127': 'specRepresentation', 'MS:1000128': 'specRepresentation',
#                      'MS:1000505': 'MaxIntensity', 'MS:1000285': 'SumIntensity',
#                      'MS:1000016': 'Rt'
#                      })
#         self.df.columns = [self.obo_ids[x]['name'].replace(' ', '_') if x in self.obo_ids.keys() else x for x in
#                            self.df.columns.values]
#         self.df['fname'] = self.fpath
#
#     def stime_conv(self):
#         self.df['Rt'] = self.df['Rt'].astype(float)
#         if 'min' in self.df['time_unit']:
#             self.df['Rt']=self.df['Rt']*60
#             self.df['time_unit'] = 'sec'
#
# class MsExp(MSstat, Fdet):
#
#
#     @classmethod
#     def mzml(cls, fpath:str, mslev:str='1', print_summary: bool = True):
#         # read mzml files and instatiate class with read in variabels
#         # fpath, mslev = '1', print_summary = False
#         da = ReadM(fpath, mslev, print_summary)
#         #generate the following variables
#         # self.mslevel = mslev
#         # self.dpath = dpath
#         # self.fname = fname
#         #
#         # self.ms0string = ms0string
#         # self.ms1string = ms1string
#         # self.xrawd = xrawd  # usage: self.xrawd[self.ms0string]
#         # self.dfd = dfd  # usage: self.dfd[scantype], scantype eg., '0_2_2_5_20.0'
#         # # self.df
#         # self.summary = summary
#
#
#
#         return cls(dpath=da.fpath, fname=da.fpath, mslevel=da.mslevel, ms0string=da.ms0string, ms1string=da.ms1string,
#                    xrawd=da.xrawd, dfd=da.dfd, summary=None)
#
#         # mzml read-in vars
#         #
#         # __init__
#         # self.fpath
#         #
#         # read_exp___
#         # self.flag  # mslevel
#         # self.dstype
#         # self.df
#         # self.scantime
#         # self.I
#         # self.scantime
#         # self.summary
#         # self.nd
#         # self.Xraw
#         #
#         # self.fs
#         # self.df
#
#     @classmethod
#     def bruker(cls, dpath: str, convert: bool = True, docker: dict = {'repo': 'convo:v1', 'mslevel': 0, 'smode': [0, 1, 2, 3, 4, 5], 'amode': 2, 'seg': [0, 1, 2, 3], }):
#         da = ReadB(dpath, convert, docker)
#         # __init__
#         # self.mslevel
#         # self.smode
#         # self.docker
#         # self.dpath
#         # self.dbfile
#         # self.fname
#         # self.msmfile
#         #
#         # self.msZeroPP___
#         # self.ms0string
#         # self.ms1string
#         #
#         # rawd___
#         # self.xrawd
#         # self.dfd
#         #
#         # self.read_mm8___
#         # self.df
#         # self.summary
#
#         return cls(dpath = da.dpath, fname=da.fname, mslevel=da.mslevel, ms0string=da.ms0string, ms1string=da.ms1string, xrawd=da.xrawd, dfd=da.dfd, summary=da.summary)
#
#     @classmethod
#     def rM(cls, da):
#         return cls(dpath=da.fpath, fname=da.fpath, mslevel=da.mslevel, ms0string=da.ms0string, ms1string=da.ms1string,
#                    xrawd=da.xrawd, dfd=da.dfd, summary=None)
#
#     @classmethod
#     def rB(cls,da):
#         return cls(dpath=da.dpath, fname=da.fname, mslevel=da.mslevel, ms0string=da.ms0string, ms1string=da.ms1string,
#                    xrawd=da.xrawd, dfd=da.dfd, summary=da.summary)
#
#
#     def __init__(self, dpath, fname, mslevel, ms0string, ms1string, xrawd, dfd, summary):
#         self.mslevel = mslevel
#         self.dpath = dpath
#         self.fname = fname
#
#         self.ms0string = ms0string
#         self.ms1string = ms1string
#         self.xrawd = xrawd # usage: self.xrawd[self.ms0string]
#         self.dfd = dfd # usage: self.dfd[scantype], scantype eg., '0_2_2_5_20.0'
#         # self.df
#         self.summary = summary
#
#
#
#     # @log
#     @staticmethod
#     def window_mz_rt(Xr: np.ndarray, selection: dict = {'mz_min': None, 'mz_max': None, 'rt_min': None, 'rt_max': None},
#                      allow_none: bool = False, return_idc: bool = False):
#         part_none = any([x == None for x in selection.values()])
#         all_fill = any([x != None for x in selection.values()])
#         if not allow_none:
#             if not all_fill:
#                 raise ValueError('Define plot boundaries with selection argument.')
#         if part_none:
#             if selection['mz_min'] is not None:
#                 t1 = Xr[1] >= selection['mz_min']
#             else:
#                 t1 = np.ones(Xr.shape[1], dtype=bool)
#             if selection['mz_max'] is not None:
#                 t2 = Xr[1] <= selection['mz_max']
#             else:
#                 t2 = np.ones(Xr.shape[1], dtype=bool)
#             if selection['rt_min'] is not None:
#                 t3 = Xr[3] > selection['rt_min']
#             else:
#                 t3 = np.ones(Xr.shape[1], dtype=bool)
#             if selection['rt_max'] is not None:
#                 t4 = Xr[3] < selection['rt_max']
#             else:
#                 t4 = np.ones(Xr.shape[1], dtype=bool)
#         else:
#             t1 = Xr[1] >= selection['mz_min']
#             t2 = Xr[1] <= selection['mz_max']
#             t3 = Xr[3] > selection['rt_min']
#             t4 = Xr[3] < selection['rt_max']
#         idx = np.where(t1 & t2 & t3 & t4)[0]
#         if return_idc:
#             return idx
#         else:
#             return Xr[..., idx]
#
#     @staticmethod
#     def tick_conv(X):
#         V = X / 60
#         return ["%.2f" % z for z in V]
#
#     def _noiseT(self, p, X=None, local=True):
#         if local:
#             self.noise_thres = np.quantile(X, q=p)
#         else:
#             self.noise_thres = np.quantile(self.xrawd[self.ms0string], q=p)
#
#         idxN = X[2] > self.noise_thres
#         idc_above = np.where(idxN)[0]
#         idc_below = np.where(~idxN)[0]
#         return (idc_below, idc_above)
#
#     # @log
#     @staticmethod
#     def get_density(X: np.ndarray, bw: float, q_noise: float = 0.5):
#         xmin = X[1].min()
#         xmax = X[1].max()
#         b = np.linspace(xmin, xmax, 500)
#         idxf = np.where(X[2] > np.quantile(X[2], q_noise))[0]
#         x_rev = X[1, idxf]
#         kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(x_rev[:, np.newaxis])
#         log_dens = np.exp(kde.score_samples(b[:, np.newaxis]))
#         return (b, log_dens / np.max(log_dens))
#
#     # @log
#     def vizd(self, q_noise: float = 0.50,
#              selection: dict = {'mz_min': None, 'mz_max': None, 'rt_min': None, 'rt_max': None},
#              qcm_local: bool = True):
#         # viz 2D with density
#         # @log
#         def on_lims_change(event_ax: plt.axis):
#             x1, x2 = event_ax.get_xlim()
#             y1, y2 = event_ax.get_ylim()
#             selection = {'mz_min': y1, 'mz_max': y2, 'rt_min': x1, 'rt_max': x2}
#             Xs = self.window_mz_rt(self.xrawd[self.ms0string], selection, allow_none=False)
#             # print(np.where(Xs[2] > self.noise_thres)[0])
#             #Xs = Xs[:,Xs[2] > self.noise_thres]
#             axs[1].clear()
#             axs[1].set_ylim([y1, y2])
#             axs[1].tick_params(labelleft=False)
#             axs[1].set_xticks([])
#             axs[1].text(1.05, 0, self.fname, rotation=90, fontsize=4, transform=axs[1].transAxes)
#             axs[1].tick_params(axis='y', direction='in')
#             if Xs.shape[0] >0:
#                 bsMin = (y2 - y1) / 200 if ((y2 - y1) / 200) > 0.0001 else 0.0001
#                 bw = bsMin
#                 y, x = self.get_density(Xs, bw, q_noise=q_noise)
#                 axs[1].plot(x, y, c='black')
#                 axs[1].annotate(f'bw: {np.round(bw, 5)}\np: {np.round(q_noise, 3)}',
#                                 xy=(1, y1 + np.min([0.1, float((y2 - y1) / 100)])), fontsize='xx-small', ha='right')
#
#
#         Xsub = self.window_mz_rt(self.xrawd[self.ms0string], selection, allow_none=False)
#         idc_below, idc_above = self._noiseT(p=q_noise, X=Xsub, local=qcm_local)
#         cm = plt.cm.get_cmap('rainbow')
#         fig, axs = plt.subplots(1, 2, sharey=False, gridspec_kw={'width_ratios': [3, 1]})
#
#         axs[1].tick_params(labelleft=False)
#         axs[1].set_xticks([])
#         axs[1].text(1.05, 0, self.fname, rotation=90, fontsize=4, transform=axs[1].transAxes)
#         axs[1].tick_params(axis='y', direction='in')
#
#         if len(idc_below) > 0:
#             axs[0].scatter(Xsub[3, idc_below], Xsub[1, idc_below], s=0.1, c='gray', alpha=0.5)
#
#         if len(idc_above) > 0:
#             im = axs[0].scatter(Xsub[3, idc_above], Xsub[1, idc_above], c=Xsub[2, idc_above], s=5, cmap=cm,
#                                 norm=LogNorm())
#             cbaxes = fig.add_axes([0.15, 0.77, 0.021, 0.1])
#             cb = fig.colorbar(mappable=im, cax=cbaxes, orientation='vertical',
#                               format=LogFormatterSciNotation(base=10, labelOnlyBase=False))
#             cb.ax.tick_params(labelsize='xx-small')
#             axs[0].callbacks.connect('ylim_changed', on_lims_change)
#         fig.subplots_adjust(wspace=0.05)
#         fig.show()
#         return (fig, axs)
#
#     # @log
#     def viz(self, q_noise: float = 0.89,
#             selection: dict = {'mz_min': None, 'mz_max': None, 'rt_min': None, 'rt_max': None}, qcm_local: bool = True):
#         # viz 2D
#         Xsub = self.window_mz_rt(self.xrawd[self.ms0string], selection, allow_none=False)
#         idc_below, idc_above = self._noiseT(p=q_noise, X=Xsub, local=qcm_local)
#         cm = plt.cm.get_cmap('rainbow')
#         fig = plt.figure()
#         ax = fig.add_subplot(111)
#         ax.text(1.25, 0, self.dpath, rotation=90, fontsize=6, transform=ax.transAxes)
#
#         if len(idc_below) > 0:
#             ax.scatter(Xsub[3, idc_below], Xsub[1, idc_below], s=0.1, c='gray', alpha=0.5)
#
#         if len(idc_above) > 0:
#             im = ax.scatter(Xsub[3, idc_above], Xsub[1, idc_above], c=Xsub[2, idc_above], s=5, cmap=cm, norm=LogNorm())
#             fig.colorbar(im, ax=ax)
#         fig.canvas.draw()
#         ax.set_xlabel(r"$\bfScan time$ [sec]")
#         ax2 = ax.twiny()
#         ax.yaxis.offsetText.set_visible(False)
#         ax.yaxis.set_label_text(r"$\bfm/z$")
#         tmax = np.max(Xsub[3])
#         tmin = np.min(Xsub[3])
#         tick_loc = np.linspace(tmin / 60, tmax / 60, 5) * 60
#         ax2.set_xlim(ax.get_xlim())
#         ax2.set_xticks(tick_loc)
#         ax2.set_xticklabels(self.tick_conv(tick_loc))
#         ax2.set_xlabel(r"[min]")
#         # fig.colorbar(im, ax=ax)
#         fig.show()
#         return (fig, ax)
#
#     def massSpectrumDIA(self, rt: float = 150., scantype: str = '0_2_2_5_20.0', viz: bool = False):
#         import matplotlib.pyplot as plt
#         import numpy as np
#         df = self.dfd[scantype]
#         idx = np.argmin(np.abs(df.Rt - rt))
#         inf = df.iloc[idx]
#         sub = self.xrawd[scantype]
#         sub = sub[..., sub[0] == (inf.Id - 1)]
#         if viz:
#             f, ax = plt.subplots(1, 1)
#             mz = sub[1]
#             intens = sub[2]
#             ax.vlines(sub[1], np.zeros_like(mz), intens / np.max(intens) * 100)
#             ax.text(0.77, 0.95, f'Acq Type: {inf.LevPP.upper()}', rotation=0, fontsize=8, transform=ax.transAxes,
#                     ha='left')
#             ax.text(0.77, 0.9, f'Rt: {inf.Rt} s', rotation=0, fontsize=8, transform=ax.transAxes, ha='left')
#             ax.text(0.77, 0.85, f'Collision E: {inf.Collision_Energy_Act} eV', rotation=0, fontsize=8,
#                     transform=ax.transAxes, ha='left')
#             return (f, ax)
#         else:
#             return (sub[1], sub[2], np.zeros_like(sub[0]), inf)
#
#     def ms2L(self, q_noise, selection, qcm_local: bool = True):
#         mpl.rcParams['keymap.back'].remove('left') if ('left' in mpl.rcParams['keymap.back']) else None
#
#         def on_lims_change(event_ax: plt.axis):
#             x1, x2 = event_ax.get_xlim()
#             y1, y2 = event_ax.get_ylim()
#             selection = {'mz_min': y1, 'mz_max': y2, 'rt_min': x1, 'rt_max': x2}
#             Xs = self.window_mz_rt(self.xrawd[self.ms0string], selection, allow_none=False)
#             bsMin = (y2 - y1) / 200 if ((y2 - y1) / 200) > 0.0001 else 0.0001
#             bw = bsMin
#             y, x = self.get_density(Xs, bw, q_noise=q_noise)
#             axs[1, 1].clear()
#             axs[1, 1].set_ylim([y1, y2])
#             axs[1, 1].plot(x, y, c='black')
#             axs[1, 1].tick_params(labelleft=False)
#             axs[1, 1].set_xticks([])
#             axs[1, 1].annotate(f'bw: {np.round(bw, 5)}\np: {np.round(q_noise, 3)}',
#                                xy=(1, y1 + np.min([0.1, float((y2 - y1) / 100)])), fontsize='xx-small', ha='right')
#             axs[1, 1].text(1.05, 0, self.fname, rotation=90, fontsize=4, transform=axs[1, 1].transAxes)
#             axs[1, 1].tick_params(axis='y', direction='in')
#
#         def on_key(event_ax: plt.axis):
#             self.xind.set_visible(False)
#             xl = axs[0, 0].get_xlim()
#             # xu = axs[0, 0].get_ylim()
#
#             if event_ax.key == 'f':
#                 self.rtPlot = event_ax.xdata
#                 axs[0, 0].clear()
#                 mz, intens, ylim0, inf = self.massSpectrumDIA(rt=self.rtPlot, scantype=self.ms1string, viz=False)
#
#             if event_ax.key == 'right':
#                 self.rtPlot = self.dfd[self.ms1string].iloc[
#                     np.argmin(np.abs(self.rtPlot - self.dfd[self.ms1string].Rt)) + 1].Rt
#                 xl = axs[0, 0].get_xlim()
#                 axs[0, 0].clear()
#                 mz, intens, ylim0, inf = self.massSpectrumDIA(rt=self.rtPlot, scantype=self.ms1string, viz=False)
#                 axs[0, 0].set_xlim(xl)
#
#             if event_ax.key == 'left':
#                 self.rtPlot = self.dfd[self.ms1string].iloc[
#                     np.argmin(np.abs(self.rtPlot - self.dfd[self.ms1string].Rt)) - 1].Rt
#                 xl = axs[0, 0].get_xlim()
#                 axs[0, 0].clear()
#                 mz, intens, ylim0, inf = self.massSpectrumDIA(rt=self.rtPlot, scantype=self.ms1string, viz=False)
#                 axs[0, 0].set_xlim(xl)
#
#             axs[0, 0].vlines(mz, ylim0, intens)
#             axs[0, 0].text(0.77, 0.85, f'Acq Type: {inf.LevPP.uppder() if inf.LevPP is not None else "-"}', rotation=0,
#                            fontsize='xx-small',
#                            transform=axs[0, 0].transAxes, ha='left')
#             axs[0, 0].text(0.77, 0.75, f'Rt: {np.round(inf.Rt, 2)} s ({inf.Id})', rotation=0, fontsize='xx-small',
#                            transform=axs[0, 0].transAxes, ha='left')
#             axs[0, 0].text(0.77, 0.65, f'Collision E: {inf.Collision_Energy_Act} eV', rotation=0, fontsize='xx-small',
#                            transform=axs[0, 0].transAxes, ha='left')
#             axs[0, 0].text(0.77, 0.55, f'Parent SID: {inf.Parent}', rotation=0, fontsize='xx-small',
#                            transform=axs[0, 0].transAxes, ha='left')
#             axs[0, 0].text(0.77, 0.45, f'Isol mass: {np.round(inf.MSMS_IsolationMass_Act, 2)}', rotation=0,
#                            fontsize='xx-small',
#                            transform=axs[0, 0].transAxes, ha='left')
#             axs[0, 0].text(0.77, 0.35, f'Q Iso-Res: {np.round(inf.Quadrupole_IsolationResolution_Act, 2)}', rotation=0,
#                            fontsize='xx-small', transform=axs[0, 0].transAxes, ha='left')
#
#             d = (axs[1, 0].transData + axs[1, 0].transAxes.inverted()).transform((self.rtPlot, 100))
#             self.xind = axs[1, 0].scatter([d[0]], [-0.01], c='red', clip_on=False, marker='^',
#                                           transform=axs[1, 0].transAxes)
#
#         Xsub = self.window_mz_rt(self.xrawd[self.ms0string], selection, allow_none=False)
#         idc_below, idc_above = self._noiseT(p=q_noise, X=Xsub, local=qcm_local)
#
#         cm = plt.cm.get_cmap('rainbow')
#         fig, axs = plt.subplots(2, 2, sharey='row', gridspec_kw={'width_ratios': [3, 1], 'height_ratios': [1.5, 3]})
#         axs[1, 1].text(1.05, 0, self.dpath, rotation=90, fontsize=4, transform=axs[1, 1].transAxes)
#         axs[1, 1].tick_params(axis='y', direction='in')
#         axs[1, 0].scatter(Xsub[3, idc_below], Xsub[1, idc_below], s=0.1, c='gray', alpha=0.5)
#         im = axs[1, 0].scatter(Xsub[3, idc_above], Xsub[1, idc_above], c=(Xsub[2, idc_above]), s=5, cmap=cm,
#                                norm=LogNorm())
#         cbaxes = fig.add_axes([0.15, 0.44, 0.021, 0.1])
#         cb = fig.colorbar(mappable=im, cax=cbaxes, orientation='vertical',
#                           format=LogFormatterSciNotation(base=10, labelOnlyBase=False))
#         cb.ax.tick_params(labelsize='xx-small')
#         fig.subplots_adjust(wspace=0.05)
#         axs[1, 0].callbacks.connect('ylim_changed', on_lims_change)
#         axs[1, 0].callbacks.connect('ylim_changed', on_lims_change)
#         cid = fig.canvas.mpl_connect('key_press_event', on_key)
#
#         self.rtPlot = Xsub[3, np.argmax(Xsub[2])]
#         mz, intens, ylim0, inf = self.massSpectrumDIA(rt=self.rtPlot, scantype=self.ms1string, viz=False)
#         # self.intensmax = np.max(intens)
#         axs[0, 0].vlines(mz, ylim0, intens)
#         axs[0, 0].text(0.75, 0.85, f'Acq Type: {inf.LevPP.uppder() if inf.LevPP is not None else "-"}', rotation=0,
#                        fontsize='xx-small', transform=axs[0, 0].transAxes, ha='left')
#         axs[0, 0].text(0.75, 0.75, f'Rt: {inf.Rt} s ({inf.Id})', rotation=0, fontsize='xx-small',
#                        transform=axs[0, 0].transAxes, ha='left')
#         axs[0, 0].text(0.75, 0.65, f'Collision E: {inf.Collision_Energy_Act} eV', rotation=0, fontsize='xx-small',
#                        transform=axs[0, 0].transAxes, ha='left')
#         axs[0, 0].text(0.75, 0.55, f'Parent SID: {inf.Parent}', rotation=0, fontsize='xx-small',
#                        transform=axs[0, 0].transAxes, ha='left')
#         axs[0, 0].text(0.75, 0.45, f'Isol mass: {np.round(inf.MSMS_IsolationMass_Act, 2)}', rotation=0,
#                        fontsize='xx-small', transform=axs[0, 0].transAxes, ha='left')
#         axs[0, 0].text(0.75, 0.35, f'Q Iso-Res: {np.round(inf.Quadrupole_IsolationResolution_Act, 2)}', rotation=0,
#                        fontsize='xx-small', transform=axs[0, 0].transAxes, ha='left')
#         axs[0, 0].tick_params(axis='y', direction='out')
#         axs[0, 1].set_axis_off()
#
#         axs[0, 0].xaxis.set_label_text(r"$\bfm/z$")
#         axs[1, 0].set_xlabel(r"$\bfScan time$ [sec]")
#         axs[1, 0].yaxis.set_label_text(r"$\bfm/z$")
#
#         d = (axs[1, 0].transData + axs[1, 0].transAxes.inverted()).transform((self.rtPlot, 100))
#         self.xind = axs[1, 0].scatter([d[0]], [-0.01], c='red', clip_on=False, marker='^',
#                                       transform=axs[1, 0].transAxes)
#         fig.subplots_adjust(wspace=0.05, hspace=0.3)
#
#     # @log
#     def chromg(self, tmin: float = None, tmax: float = None, ctype: list = ['tic', 'bpc', 'xic'], xic_mz: float = [],
#                xic_ppm: float = 10):
#         df_l1 = self.dfd[self.ms0string]
#         if df_l1.shape[0] == 0: raise ValueError('mslevel 1 in df does not exist or is not defined')
#         xic = False
#         tic = False
#         bpc = False
#         if not isinstance(ctype, list):
#             ctype = [ctype]
#         if any([x in 'xic' for x in ctype]):
#             xic = True
#         if any([x in 'bpc' for x in ctype]):
#             bpc = True
#         if any([x in 'tic' for x in ctype]):
#             tic = True
#         if ((xic == False) & (tic == False) & (bpc == False)):
#             raise ValueError('ctype error: define chromatogram type(s) as tic, bpc or xic (string or list of strings).')
#         if isinstance(tmin, type(None)):
#             tmin = df_l1.Rt.min()
#         if isinstance(tmax, type(None)):
#             tmax = df_l1.Rt.max()
#         fig = plt.figure()
#         ax1 = fig.add_subplot(111)
#         if xic:
#             xx, xy = self.xic(xic_mz, xic_ppm, rt_min=tmin, rt_max=tmax)
#             ax1.plot(xx, xy.astype(float), label='XIC ' + str(xic_mz) + 'm/z')
#         if tic:
#             ax1.plot(df_l1.Rt, df_l1.SumIntensity.astype(float), label='TIC')
#         if bpc:
#             ax1.plot(df_l1.Rt, df_l1.MaxIntensity.astype(float), label='BPC')
#         ax1.text(1.04, 0, self.fname, rotation=90, fontsize=6, transform=ax1.transAxes)
#         fig.canvas.draw()
#         offset = ax1.yaxis.get_major_formatter().get_offset()
#         ax1.set_xlabel(r"$\bfScantime (s)$")
#         ax1.set_ylabel(r"$\bfCount$")
#         ax2 = ax1.twiny()
#         ax1.yaxis.offsetText.set_visible(False)
#         if offset != '': ax1.yaxis.set_label_text(r"$\bfInt/count$ ($x 10^" + offset.replace('1e', '') + '$)')
#         tick_loc = np.arange(np.round(tmin / 60), np.round(tmax / 60), 2, dtype=float) * 60
#         ax2.set_xlim(ax1.get_xlim())
#         ax2.set_xticks(tick_loc)
#         ax2.set_xticklabels(self.tick_conv(tick_loc))
#         ax2.set_xlabel(r"$\bfScantime (min)$")
#         ax1.legend(loc='best', bbox_to_anchor=(0.5, 0.5, 0.5, 0.5))
#         fig.show()
#         return fig, ax1, ax2
#
#     # @log
#     def d_ppm(self, mz: float, ppm: float):
#         d = (ppm * mz) / 1e6
#         return mz - (d / 2), mz + (d / 2)
#
#     # @log
#     def xic(self, mz: float, ppm: float, rt_min: float = None, rt_max: float = None):
#         mz_min, mz_max = self.d_ppm(mz, ppm)
#
#         idx_mz = np.where((self.xrawd[self.ms0string][1] >= mz_min) & (self.xrawd[self.ms0string] <= mz_max))[0]
#         if len(idx_mz) < 0:
#             raise ValueError('mz range not found')
#
#         X_mz = self.xrawd[self.ms0string][..., idx_mz]
#
#         sid = np.array(np.unique(X_mz[0]).astype(int))
#         xic = np.zeros(int(np.max(self.xrawd[self.ms0string][0]) + 1))
#         for i in sid:
#             xic[i - 1] = np.sum(X_mz[np.where(X_mz[:, 0] == i), 2])
#         stime = np.sort(np.unique(self.xrawd[self.ms0string][3]))
#
#         if (~isinstance(rt_min, type(None)) | ~isinstance(rt_max, type(None))):
#             idx_rt = np.where((stime >= rt_min) & (stime <= rt_max))[0]
#             if len(idx_rt) < 0:
#                 raise ValueError('rt range not found')
#             stime = stime[idx_rt]
#             xic = xic[idx_rt]
#
#         return (stime, xic)
#
#     @staticmethod
#     def fwhmBound(intens, st, rtDelta=1 - 0.4):
#         idxImax = np.argmax(intens)
#         ifwhm = intens[idxImax] / 2
#
#         if (idxImax == 0) or (idxImax == len(intens)):
#             return None
#
#         st_imax = st[idxImax]
#
#         ll = np.max(st[(intens < ifwhm) & (st <= st_imax)])
#         ul = np.min(st[(intens > ifwhm) & (st >= st_imax)])
#         ll1 = st_imax - ((ul - ll) * rtDelta) / 2
#         ul1 = st_imax + ((ul - ll) * rtDelta) / 2
#
#         return (ul1, ll1)
#
#     def igr(self):
#         self.l3toDf()
#         au = self.l3Df
#         rtd = 0.6  # fwhm proportion
#         fset = set(au.index.values)
#         fgr = []
#         c = 0
#         for i in fset:
#             intens = self.feat[i]['fdata']['I_sm_bline']
#             st = self.feat[i]['fdata']['st_sm']
#             out = self.fwhmBound(intens, st, rtDelta=rtd)
#
#             if out is None:
#                 continue
#
#             sub = au[(au.rt_maxI < out[0]) & (au.rt_maxI > out[1])]
#             if sub.shape[0] <= 1:
#                 continue
#
#             sidx = sub.index[np.argmax(sub.smbl.values)]
#             intens1 = self.feat[sidx]['fdata']['I_sm_bline']
#             st1 = self.feat[sidx]['fdata']['st_sm']
#             out1 = self.fwhmBound(intens1, st1, rtDelta=rtd)
#             sub1 = au[(au.rt_maxI < out1[0]) & (au.rt_maxI > out1[1])].copy()
#             sub1['fgr'] = c
#             sub1 = sub1.sort_values('mz_maxI')
#             c += 1
#
#             if any(sub1['smbl'] > 1e6):
#                 fgr.append(sub1)
#             fset = fset - set(sub1.index.values)
#
#         return pd.concat(fgr)  # candidate isotopologues
#
#
#
#
