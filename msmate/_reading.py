
# msmate, v2.0.1
# author: T Kimhofer
# Murdoch University, July 2022

# dependency import
from typeguard import typechecked
import pandas as pd
import numpy as np
import re
import xml.etree.cElementTree as ET
import os
import docker as dock
import time
import pickle
import functools
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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
# import platform
# import psutil
from typing import Union
# import sys
# import collections
# import itertools

from msmate.helpers import _collect_spectra_chrom, _get_obo, _children, _node_attr_recurse

__all__ = ['ReadB',
           'ReadM',
           'MsExp',]
#
# logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, filename='msmate.log', encoding='utf-8', format='%(asctime)s %(levelname)s %(name)s %(message)s', filemode='w')
# logger = logging.getLogger(__name__)
# logger.debug(f'{[os.cpu_count(), platform.uname(), psutil.net_if_addrs(), psutil.users()]}')
#
# def log(f):
#     @functools.wraps(f)
#     def wr(*arg, **kwargs):
#         try:
#             logger.info(f'calling {f.__name__}')
#             out = f(*arg, **kwargs)
#             logger.info(f'done {f.__name__}')
#             return out
#         except Exception as e:
#             logger.exception(f'{f.__name__}: {repr(e)}')
#     return wr
#
# import inspect
# def logIA(f):
#     @functools.wraps(f)
#     def wr(*args, **kwargs):
#         try:
#             #print(f.__dict__)
#             func_args = inspect.signature(f).bind(*args, **kwargs).arguments
#             func_args_str = ", ".join(map("{0[0]} = {0[1]!r}".format, func_args.items()))
#             logger.info(f"{f.__qualname__} ({func_args_str})")
#             out = f(*args, **kwargs)
#             #logger.info(f'done {f.__name__}')
#             return out
#
#         except Exception as e:
#             logger.exception(f'{repr(e)}')
#     return wr

# @logIA
@typechecked
class ReadB():
    """Bruker experiment class and import methods"""
    def __init__(self, dpath: str, convert: bool = True,
                 docker: dict = {'repo': 'convo:v1', 'mslevel': 0, 'smode': [0, 1, 2, 3, 4, 5], 'amode': 2,
                                 'seg': [0, 1, 2, 3], }):
        """Import of XC-MS level 1 scan data from raw Bruker or msmate data format (.d or binary `.p`, respectively)

           Note that this class is designed for use with `MsExp`.

           Args:
               dpath: Path `string` variable pointing to Bruker experiment folder (.d)
               convert: `Bool` indicating if conversion should take place using docker container inf `.p` is not present
               docker: `dict` with docker information: `{'repo': 'convo:v1', 'mslevel': 0, 'smode': [0, 1, 2, 3, 4, 5], 'amode': 2, 'seg': [0, 1, 2, 3], }`
        """
        self.mslevel = str(docker['mslevel']) if not isinstance(docker['mslevel'], list) else tuple(
            map(str, np.unique(docker['mslevel'])))
        if len(self.mslevel) == 1:
            self.mslevel = self.mslevel[0]
        self.smode = str(docker['smode']) if not isinstance(docker['smode'], list) else tuple(
            map(str, np.unique(docker['smode']).tolist()))
        if len(self.smode) == 1:
            self.smode = self.smode[0]
        self.amode = str(docker['amode'])
        self.seg = str(docker['seg']) if not isinstance(docker['seg'], list) else tuple(
            map(str, np.unique(docker['seg'])))
        if len(self.seg) == 1:
            self.seg = self.seg[0]
        self.docker = {'repo': docker['repo'], 'mslevel': self.mslevel, 'smode': self.smode, 'amode': self.amode,
                       'seg': self.seg, }
        self.dpath = os.path.abspath(dpath)
        self.dbfile = os.path.join(dpath, 'analysis.sqlite')
        self.fname = os.path.basename(dpath)
        self.msmfile = os.path.join(dpath,
                                    f'mm8v3_edata_msl{self.docker["mslevel"]}_sm{"".join(self.docker["smode"])}_am{self.docker["amode"]}_seg{"".join(self.docker["seg"])}.p')
        if os.path.exists(self.msmfile):
            self._read_mm8()
        else:
            if convert:
                self._convoDocker()
                self._read_mm8()
            else:
                raise SystemError('d folder requries conversion, enable docker convo')
        # self.ecdf()

    def _convoDocker(self):
        """Convert Bruker 2D MS experiment data to msmate file/obj using a custom build Docker image"""
        t0 = time.time()
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

    def _msZeroPP(self):
        """Identify MS acquisition/scan type and level"""
        if any((self.df.ScanMode == 2) & ~(self.df.MsLevel == 1)):  # dda
            raise ValueError('Check AcquisitionKeys for scan types - combo not encountered before')

        if any((self.df.ScanMode == 5) & ~(self.df.MsLevel == 0)):  # dia (bbcid)
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
        add = self.df['LevPP'].copy()
        add.loc[idx_dda] = 'dda'
        add.loc[idx_dia] = 'dia'
        add.loc[idx_ms0] = 'fs'
        self.df['LevPP'] = add

    def _rawd(self, ss):
        """Organise raw MS data and scan metadata according to ms level and scan/acquisition type."""
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

    def _read_mm8(self):
        """Import msmate data, determine scan/acquisition/mode types and create characteristic set of data objects"""
        ss = pickle.load(open(self.msmfile, 'rb'))
        self.df = pd.DataFrame(ss['sinfo'])
        # define ms level for viz and peak picking
        idcS3 = (self.df.Segment == 3).values
        if any(idcS3):
            self.df['stype'] = "0"
            add = self.df['stype'].copy()
            add[idcS3] = [
                f'{x["MsLevel"]}_{x["AcquisitionKey"]}_{x["AcquisitionMode"]}_{x["ScanMode"]}_{x["Collision_Energy_Act"]}'
                for x in ss['sinfo'] if x['Segment'] == 3]
            self.df['stype'] = add

        self._msZeroPP()
        # create dict for each scantype and df
        self._rawd(ss)
        polmap = {0: 'P', 1: 'N'}
        df1 = pd.DataFrame(
            [(polmap[x['Polarity']], str(x['MzAcqRangeLower']) + '-' + str(x['MzAcqRangeUpper']), x['MsLevel']) for x in
             ss['sinfo']], columns=['Polarity', 'MzAcqRange', 'MsLevel'])
        df1 = pd.DataFrame([self._csummary(i, df1) for i in range(df1.shape[1])], index=df1.columns).transpose()
        df1['nSc'] = len(ss['sinfo'])
        df2 = pd.DataFrame([(x['AcquisitionMode'], x['ScanMode'], x['Segment']) for x in ss['sinfo']],
                           columns=['AcqMode', 'ScMode', 'Segments'])
        t = pd.DataFrame(df2.groupby(['Segments']).apply(lambda x: self._vc(x, n=df2.shape[0])))
        t = t.rename(index={t.index[i]: 'Segment ' + str(t.index[i]) for i in range(t.shape[0])}).transpose()
        self.summary = pd.concat([df1, t], axis=1)


    @staticmethod
    def _csummary(i: int, df: pd.DataFrame):
        """Summary function for scan metadata information."""
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
    def _vc(x, n):
        """Generate unique combinations of scan acquisition types/modes."""
        import numpy as np
        ct = x.value_counts(subset=['ScMode', 'AcqMode'])
        s = []
        for i in range(ct.shape[0]):
            s.append(
                f'{ct.index.names[0]}={ct.index[i][0]} & {ct.index.names[1]}={ct.index[i][1]}: {ct[ct.index[i]]} ({np.round(ct[ct.index[0]] / n * 100, 1)} %)')
        return '; '.join(s)

# @logIA
@typechecked
class ReadM():
    """mzML experiment class and import methods"""
    def __init__(self, fpath: str, mslev:str='1'):
        """Import of XC-MS level 1 scan data from mzML files V1.1.0

            Note that this class is designed for use with `MsExp`.

            Args:
                fpath: Path `string` variable pointing to mzML file
                mslev: `String` defining ms level read-in (fulls can)
        """
        self.fpath = fpath
        if mslev != '1':
            raise ValueError('Check mslev argument - This function currently support ms level 1 import only.')
        self.mslevel = mslev
        self._read_mzml()
        self._createSpectMat()
        self._stime_conv()
        self.df['subsets'] = self.df.ms_level+self.df.scan_polarity
        ids = np.unique(self.df.subsets, return_counts=True)
        imax = np.argmax(ids[1])
        self.ms0string = ids[0][imax]
        self.ms1string = None
        if len(ids[0]) > 1:
            print(f'Experiment done with polarity switching! Selecting {ids[0][imax]} as default level 1 data set. Alter attribute `ms0string` to change polarity')
            self.xrawd = {}
            self.dfd = {}
            print(self.Xraw.shape)
            Xr = pd.DataFrame(self.Xraw.T)
            Xr.index=self.Xraw[0]
            for i, d in enumerate(ids[0]):
                self.dfd.update({d: self.df[self.df.subsets == d]})
                self.xrawd.update({d: Xr.loc[self.dfd[d]['index'].astype(int).values].to_numpy().T})
        else:
            self.xrawd = {self.ms0string: self.Xraw}
            self.dfd = {self.ms0string: self.df}

    def _read_mzml(self):
        """Extracts MS data from mzml file"""
        # this is for mzml version 1.1.0
        # schema specification: https://raw.githubusercontent.com/HUPO-PSI/mzML/master/schema/schema_1.1/mzML1.1.0.xsd
        # read in data, files index 0 (see below)
        # flag is 1/2 for msLevel 1 or 2
        tree = ET.parse(self.fpath)
        root = tree.getroot()
        child = _children(root)
        imzml = child.index('mzML')
        mzml_children = _children(root[imzml])
        obos = root[imzml][mzml_children.index('cvList')] # controlled vocab (CV
        self.obo_ids = _get_obo(obos, obo_ids={})
        seq = np.where(~np.isin(mzml_children, ['cvList', 'run']))[0]
        pp = {}
        for j in seq:
            filed = _node_attr_recurse(s=root[imzml][j], d=4, c=0, ii=[])
            dn = {}
            for i in range(len(filed)):
                dn.update(dict(zip([filed[i]['path'] + '_' + re.sub('\{.*\}', '', x) for x in list(filed[i].keys())[1:]],
                                   list(filed[i].values())[1:])))
            pp.update({mzml_children[j]: dn})
        run = root[imzml][mzml_children.index('run')]
        self.out31 = _collect_spectra_chrom(s=run, ii={}, d=20, c=0, flag=self.mslevel, tag='', obos=self.obo_ids)

    def _createSpectMat(self):
        """Organise raw MS data and scan metadata"""

        sc_msl1 = [i for i, x in self.out31.items() if len(x['data']) == 2]  # sid of ms level 1 scans
        # summarise and collect data
        ms = []
        meta = []
        sid_count = 0
        for i in sc_msl1:
            # if ('s' in i) : # and (d['meta']['MS:1000511'] == '1')   # scans, obo ms:1000511 -> ms level
            d = self.out31[i]
            meta.append(d['meta'])
            add = []
            for j, dd in (d['data']).items():
                add.append(dd['d'])
            idm = np.ones_like(dd['d'])
            add.append(idm * sid_count)
            add.append(idm * float(d['meta']['MS:1000016']))  # obo ms:1000016 -> scan start time
            ms.append((i, np.array(add)))
            sid_count += 1

        Xraw = np.concatenate([x[1] for x in ms], axis=1)

        # 0: scanIdOri,
        # 1: mz,
        # 2: Int,
        # 3: st,
        # 4: scanIdNorm,

        self.Xraw = np.array([Xraw[2], Xraw[0], Xraw[1], Xraw[3], Xraw[2]])
        self.df = pd.DataFrame(meta, index=sc_msl1)
        pscan = np.repeat('None', self.df.shape[0])
        if 'MS:1000129' in self.df.columns:
            pscan[self.df['MS:1000129'].values == True] = "N"
        if 'MS:1000130' in self.df.columns:
            pscan[self.df['MS:1000130'].values == True] = 'P'

        self.df['scan_polarity'] = pscan

        if 'MS:1000127' in self.df.columns:
            add = np.repeat('centroided', self.df.shape[0])
            add[~(self.df['MS:1000127'] == True)] = 'profile'
            self.df['MS:1000127'] = add
        if 'MS:1000128' in self.df.columns:
            add = np.repeat('profile', self.df.shape[0])
            add[~(self.df['MS:1000128'] == True)] = 'centroided'
            self.df['MS:1000128'] = add

        self.df = self.df.rename(
            columns={'UO:0000010': 'time_unit', 'UO:0000031': 'time_unit', 'defaultArrayLength': 'n',
                     'MS:1000129': 'polNeg', 'MS:1000130': 'polPos',
                     'MS:1000127': 'specRepresentation', 'MS:1000128': 'specRepresentation',
                     'MS:1000505': 'MaxIntensity', 'MS:1000285': 'SumIntensity',
                     'MS:1000016': 'Rt'
                     })
        self.df.columns = [self.obo_ids[x]['name'].replace(' ', '_') if x in self.obo_ids.keys() else x for x in
                           self.df.columns.values]
        self.df['fname'] = self.fpath

        # sc_msl1 = [i for i, x in self.out31.items() if len(x['data']) == 2] # sid of ms level 1 scans
        # # ms = collections.defaultdict(list)
        # meta = collections.defaultdict(list)
        # N = collections.defaultdict(list)
        # P = collections.defaultdict(list)
        # for i in sc_msl1:
        #     # if ('s' in i) : # and (d['meta']['MS:1000511'] == '1')   # scans, obo ms:1000511 -> ms level
        #     d = self.out31[i]
        #     mz = d['data']['m/z']['d']
        #     ilen = len(mz)
        #     # st = [float(d['meta']['MS:1000016'])] * ilen
        #     # sid = [int(d['meta']['index']) + 1] * ilen
        #
        #     if 'MS:1000129' in d['meta'].keys():
        #         meta['N'].append(d['meta'])
        #         # sid_norm = [len(meta['N'])] * ilen
        #         #
        #         # N['sid'] = N['sid'] + sid
        #         # N['mz'] = N['mz'] + mz
        #         # N['int_'] = N['int_'] + d['data']['Int']['d']
        #         # N['st'] = N['st'] + st
        #         # N['sid_norm'] = N['sid_norm'] + sid_norm
        #
        #         # N['sid'].append(sid)
        #         N['n'].append(ilen)
        #         N['mz'].append(mz)
        #         N['int_'].append(d['data']['Int']['d'])
        #         # N['st'].append(st)
        #         # N['sid_norm'].append(st)
        #
        #     elif 'MS:1000130' in d['meta'].keys():
        #         meta['P'].append(d['meta'])
        #         sid_norm = (idm * len(meta['P'])).astype(int)
        #         dar = (sid, mz, d['data']['Int']['d'], st, sid_norm)
        #         P['sid'].append(sid)
        #         P['mz'].append(mz)
        #         P['int_'].append(d['data']['Int']['d'])
        #         P['st'].append(st)
        #         P['sid_norm'].append(st)
        #
        #
        # ms['P'].append(dar)
        #
        # xrawd = {}
        # dfd = {}
        # for i in meta.keys():
        #     if i == 'N':
        #
        #         r=[x for xs in N['mz'] for x in xs]
        #         s=[x for xs in N['int_'] for x in xs]
        #
        #         tt=[j for i, k in enumerate(N['n']) for j in [i]*k]
        #         tt = [j for i, k in enumerate(N['n']) for j in [i] * k]
        #         tt = [j for i, k in enumerate(N['n']) for j in [i] * k]
        #
        #
        #         pd.DataFrame([list(itertools.chain.from_iterable(N['sid'])),
        #         list(itertools.chain.from_iterable(N['mz'])),
        #         list(itertools.chain.from_iterable(N['int_'])),
        #         list(itertools.chain.from_iterable(N['st'])),
        #         list(itertools.chain.from_iterable(N['sid_norm']))])
        #         add = np.concatenate([N['sid'], N['mz'], N['int_'], N['st'], N['sid_norm']])
        #         add = np.c_([N['sid'], N['mz']])
        #
        #     pd.DataFrame([list(itertools.chain.from_iterable(N['mz'])),
        #                   list(map(float, list(itertools.chain.from_iterable(N['sid']))))])
        #
        #     xrawd[i] = np.concatenate(ms[i], axis=1)
        #     dfd[i] = np.concatenate(meta[i], axis=1)
        #     dfd[i]['scan_polarity'] = i
        #     if 'MS:1000127' in dfd[i].columns:
        #         add = np.repeat('centroided', dfd[i].shape[0])
        #         add[~(dfd[i]['MS:1000127'] == True)] = 'profile'
        #         dfd[i]['MS:1000127'] = add
        #     if 'MS:1000128' in self.df.columns:
        #         add = np.repeat('profile', dfd[i].shape[0])
        #         add[~(dfd[i]['MS:1000128'] == True)] = 'centroided'
        #         dfd[i]['MS:1000128'] = add
        #
        #     dfd[i] = dfd[i].rename(
        #         columns={'UO:0000010': 'time_unit', 'UO:0000031': 'time_unit', 'defaultArrayLength': 'n',
        #                  'MS:1000129': 'polNeg', 'MS:1000130': 'polPos',
        #                  'MS:1000127': 'specRepresentation', 'MS:1000128': 'specRepresentation',
        #                  'MS:1000505': 'MaxIntensity', 'MS:1000285': 'SumIntensity',
        #                  'MS:1000016': 'Rt'
        #                  })
        #     dfd[i].columns = [self.obo_ids[x]['name'].replace(' ', '_') if x in self.obo_ids.keys() else x for x in
        #                        dfd[i].columns.values]
        #     dfd[i]['fname'] = dfd[i].fpath


    def _stime_conv(self):
        """Performs scan time conversion from minutes to seconds."""
        self.df['Rt'] = self.df['Rt'].astype(float)
        if 'min' in self.df['time_unit'].iloc[1]:
            self.df['Rt']=self.df['Rt']*60
            self.df['time_unit'] = 'sec'
            self.Xraw[3] = self.Xraw[3] * 60

@typechecked
class MSstat:
    """ Base class defining subsetting & statistical functions operating on 2D MS data.

    Note that this class is designed for use with `MsExp`.
    """
    def ecdf(self, plot:bool = False):
        """Calculates empirical cumulative density fct"""
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
        """Inverse empirical cumulative density fct"""
        p = np.array(p)[..., np.newaxis]
        if p.ndim ==1:
            p = p[np.newaxis, ...]
        if (any(p < 0)) | (any(p > 1)):
            raise ValueError('p value range: 0-1 ')
        return self.ecdf_x[np.argmin(np.abs(self.ecdf_y - p), axis=1)]

        # @log

    @staticmethod
    def _window_mz_rt(Xr: np.ndarray,
                      selection: dict = {'mz_min': None, 'mz_max': None, 'rt_min': None, 'rt_max': None},
                      allow_none: bool = False, return_idc: bool = False):
        """2D filter function using retention/scan time and mz dimension."""
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

    # @log
    def _noiseT(self, p, X=None, local=True):
        """Calculation of a noise intensity threshold using all data points or windowed data points."""
        if local:
            self.noise_thres = np.quantile(X, q=p)
        else:
            self.noise_thres = np.quantile(self.xrawd[self.ms0string], q=p)

        idxN = X[2] > self.noise_thres
        idc_above = np.where(idxN)[0]
        idc_below = np.where(~idxN)[0]
        return (idc_below, idc_above)

    # @log
    def _d_ppm(self, mz: float, ppm: float):
        d = (ppm * mz) / 1e6
        return mz - (d / 2), mz + (d / 2)

    # def calcSignalWindow(self, p: float = 0.999, plot: bool = True, sc: float =300,  **kwargs):
    #     """experimental"""
    #     Xr = self.Xraw[self.Xraw[...,2] > self.get_ecdfInt(p),...]
    #     Xs=Xr[(Xr[...,1]>300) & (Xr[...,3]<180)]
    #     t=np.histogram2d(Xs[..., 1]*sc, Xs[..., 3]) #, **kwargs
    #     extent = [t[2][0], t[2][-1], t[1][0]/sc, t[1][-1]/sc]
    #     if plot:
    #         f, axs = plt.subplots(1, 2)
    #         axs[0].matshow(t[0]/sc,  extent=extent, origin='lower', aspect='auto')
    #     ii = np.where(t[0] == np.max(t[0]))
    #     self.mzsignal = mzra = (t[1][ii[0][0]]/sc, t[1][ii[0][0]+1]/sc)
    #     self.rtsignal = rtra = (t[2][ii[1][0]]-30, t[2][ii[1][0] + 1]+30)
    #     print(self.rtsignal)
    #     print(rtra)
    #     print(f'mz range: {mzra}, rt range (s): {np.round(rtra, 0)}')
    #     idc=(self.Xraw[..., 1] > mzra[0]) & (self.Xraw[..., 1] < mzra[1]) & (self.Xraw[..., 3] > (rtra[0])) & (self.Xraw[..., 3] < (rtra[1]))
    #     self.Xsignal = self.Xraw[idc,...]
    #     self.Xsignal = self.Xsignal[self.Xsignal[...,2]> self.get_ecdfInt(0.99)]
    #     if plot:
    #         axs[1].scatter(self.Xsignal[...,3], self.Xsignal[...,1], s=0.5, c=np.log(self.Xsignal[...,2]))
    # def scans_sec(self):
    #     asps = 1/((self.scantime.max() - self.scantime.min())/self.summary.nSc.iloc[0]) # hz
    #     ii = np.diff(np.unique(self.scantime))
    #     self.summary = pd.concat([self.summary, pd.DataFrame({'ScansPerSecAver': asps, 'ScansPerSecMin': 1/ii.max(), 'ScansPerSecMax': 1/ii.min()}, index=[0])], axis=1)

@typechecked
class Fdet:
    """ Base class defining feature detection functions operating on 2D MS data.

    Note that this class is designed for use with `MsExp`.
    """
    # @log
    def getScaling(self, **kwargs):
        """Manual selection of signal trace to calculate m/z dimension scaling factor"""
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


    # @log
    def pp(self, mz_adj: float = 40, q_noise: float = 0.99, dbs_par: dict = {'eps': 1.1, 'min_samples': 3},
                     selection={'mz_min': None, 'mz_max': None, 'rt_min': None, 'rt_max': None},
                     qc_par: dict = {'st_len_min': 5, 'raggedness': 0.15, 'non_neg': 0.8, 'sId_gap': 0.15, 'sino': 2,
                             'ppm': 15}, \
                     qcm_local: bool=True)\
            :
        """Feature detection.

        Perform feature detection using dbscan and signal qc criteria.

        Features are detected in sID/mz dimensions after scaling the mz dimension and classified into four different quality categories:
        L0: All features detected
        L1: Features comprising of n or more data points (`st_len_min`)
        L2: Features that do not pass one or more of the defined quality filters (see below)
        L3: Features that pass all quality filters (see below)

        Quality filter:

            raggedness: quantifies the monotonicity before and after peak maximum (0:


            Args:
                mz_adj: Multiplicative adjustment factor for the m/z dimension (`float`)
                q_noise: Noise intensity threshold provided as quantile probability (`float`)
                dbs_par: dbscan parameters as dictionary (`eps` and `min_samples)
                selection: M/z and scantime (in sec) window
                qc_par: Feature quality filter parameters as dictionary (see Details)
                qcm_local: `bool` indicating if quantile intensity thresh. should be calculated locally in m/z and st window or globally
        """
        s0 = time.time()
        self.mz_adj = mz_adj
        self.q_noise = q_noise
        self.dbs_par = dbs_par
        self.qc_par = qc_par
        self.selection = selection

        # if desired reduce mz/rt regions for ppicking
        # Xraw has 5 rows: scanIdOri, mz, int, st, scanIdNorm
        self.Xf = self._window_mz_rt(self.xrawd[self.ms0string], self.selection, allow_none=True, return_idc=False)

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
        self._feat_summary1()
        s4 = time.time()

        print(f'total time pp (min): {np.round((s4-s0)/60, 1)}')

    def _feat_summary1(self):
        """Define feature classes based on quality constraints"""
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
            f1 = self._cls(cl_id=fid, Xf=self.Xf, qc_par=self.qc_par)
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

        print(f'Time for {len(cl_n_above)} feature summaries: {np.round(t3-t2)}s ({np.round((t3-t2)/len(cl_n_above), 2)}s per ft)')

        if (len(self.feat_l2) == 0) & (len(self.feat_l3) == 0):
            raise ValueError('No L2 features found')
        else:
            print(f'Number of L2 features: {len(self.feat_l2)+len(self.feat_l3)} ({np.round((len(self.feat_l2)+len(self.feat_l3))/len(cl_n_above)*100)}%)')

        if len(self.feat_l3) == 0:
            # raise ValueError('No L3 features found')
            print('No L3 features found')
        else:
            print(f'Number of L3 features: {len(self.feat_l3)} ({np.round(len(self.feat_l3) / (len(self.feat_l2)+len(self.feat_l3))*100,1)}%)')

    # # @background
    # def cl_summary(self, cl_id: Union[int, np.int64]):
    #     """Cluster summary"""
    #     # cl_id = cl_n_above[0]
    #     # cluster summary
    #     # X is data matrix
    #     # dbs is clustering object with lables
    #     # cl_id is cluster id in labels
    #     # st_adjust is scantime adjustment factor
    #
    #     # Xf has 8 columns:
    #     # 0: scanIdOri,
    #     # 1: mz,
    #     # 2: Int,
    #     # 3: st,
    #     # 4: scanIdNorm,
    #     # 5: noiseBool,
    #     # 6: st_adj,
    #     # 7: clMem
    #
    #     # import matplotlib.pyplot as plt
    #     # cl_id = 3660
    #     @log
    #     def wImean(x, I):
    #         # intensity weighted mean of st
    #         w = I / np.sum(I)
    #         wMean_val = np.sum(w * x)
    #         wMean_idx = np.argmin(np.abs(x - wMean_val))
    #         return (wMean_val, wMean_idx)
    #
    #     @log
    #     def ppm_mz(mz):
    #         return np.std(mz) / np.mean(mz) * 1e6
    #
    #     # dp idx for cluster
    #     idx = np.where(self.Xf[7] == cl_id)[0]
    #     iLen = len(idx)
    #
    #     # gaps or doublets in successive scan ids
    #     x_sid = self.Xf[4, idx]
    #     sid_diff = np.diff(x_sid)
    #     sid_gap = np.sum((sid_diff > 1) | (sid_diff < 1)) / (iLen - 1)
    #
    #     # import matplotlib.pyplot as plt
    #     x = self.Xf[3, idx]  # scantime
    #     y = self.Xf[2, idx]  # intensity
    #     mz = self.Xf[1, idx]  # mz
    #     ppm = ppm_mz(mz)
    #
    #     mz_min = np.min(mz)
    #     mz_max = np.max(mz)
    #     rt_min = np.min(x)
    #     rt_max = np.max(x)
    #
    #
    #     if ppm > self.qc_par['ppm']:
    #         crit = 'm/z variability of a feature (ppm)'
    #         qc = {'sId_gap': sid_gap, 'ppm': ppm}
    #         quant = {}
    #         descr = {'ndp': iLen, 'st_span': np.round(rt_max - rt_min, 1), \
    #                  'mz_min': mz_min, 'mz_max': mz_max, \
    #                  'rt_min': rt_min, 'rt_max': rt_max, }
    #         fdata = {'st': x, 'I_raw': y}
    #         od = {'flev': 2, 'crit': crit, 'qc': qc, 'quant': quant, 'descr': descr, 'fdata': fdata}
    #         return od
    #
    #     if sid_gap > self.qc_par['sId_gap']:
    #         crit = 'sId_gap: scan ids not consecutive (%)'
    #         qc = {'sId_gap': sid_gap, 'ppm': ppm}
    #         quant = {}
    #         descr = {'ndp': iLen, 'st_span': np.round(rt_max - rt_min, 1), \
    #                  'mz_min': mz_min, 'mz_max': mz_max, \
    #                  'rt_min': rt_min, 'rt_max': rt_max, }
    #         fdata = {'st': x, 'I_raw': y}
    #         od = {'flev': 2, 'crit': crit, 'qc': qc, 'quant': quant, 'descr': descr, 'fdata': fdata}
    #         return od
    #
    #     # _, icent = wImean(x, y)
    #
    #     # minor smoothing - minimum three data points - and  padding
    #     ysm = np.convolve(y, np.ones(3) / 3, mode='valid')
    #     ysm = np.concatenate(([y[0]], ysm, [y[-1]]))
    #     xsm = x
    #
    #
    #     # I min-max scaling and bline correction
    #     y_max = np.max(y)
    #     y = y / y_max
    #     ysm_max = np.max(ysm)
    #     ysm = ysm / ysm_max
    #     # bline correction of smoothed signal
    #     # if smoothed signal has less than three data points, then this won't work
    #     # minimum last points minu
    #     y_bl = ((np.min(ysm[-3:]) - np.min(ysm[0:3])) / (xsm[-1] - xsm[0])) * (xsm - np.min(xsm)) + np.min(ysm[0:3])
    #
    #     ybcor = ysm - y_bl
    #     bcor_max = np.max(ybcor * ysm_max)
    #
    #     # check peak centre
    #     _, icent = wImean(xsm, ybcor)
    #
    #     # if (icent < 2) | (icent > len(idxsm) - 2):
    #     if (icent < 2) | (icent > (len(idx) - 2)):
    #         crit = 'icent: Peak is not centered'
    #         qc = {'sId_gap': sid_gap, 'icent': icent, 'ppm': ppm}
    #         quant = {}
    #         descr = {'ndp': iLen, 'st_span': np.round(rt_max - rt_min, 1), \
    #                  'mz_min': mz_min, 'mz_max': mz_max, \
    #                  'rt_min': rt_min, 'rt_max': rt_max, }
    #         fdata = {'st': x, 'st_sm': xsm, 'I_sm_bline': ybcor * ysm_max, 'I_smooth': ysm * ysm_max,
    #                  'I_raw': y * y_max,
    #                  'I_bl': y_bl * ysm_max, }
    #         od = {'flev': 2, 'crit': crit, 'qc': qc, 'quant': quant, 'descr': descr, 'fdata': fdata}
    #         return od
    #
    #     nneg = np.sum(ybcor >= -0.05) / len(ysm)
    #     if nneg < self.qc_par['non_neg']:
    #         crit = 'nneg: Neg Intensity values'
    #         qc = {'sId_gap': sid_gap, 'non_neg': nneg, 'ppm': ppm}
    #         quant = {}
    #         descr = {'ndp': iLen, 'st_span': np.round(rt_max - rt_min, 1), \
    #                  'mz_min': mz_min, 'mz_max': mz_max, \
    #                  'rt_min': rt_min, 'rt_max': rt_max, }
    #         fdata = {'st': x, 'st_sm': xsm, 'I_sm_bline': ybcor * ysm_max, 'I_smooth': ysm * ysm_max,
    #                  'I_raw': y * y_max, 'I_bl': y_bl * ysm_max, }
    #         od = {'flev': 2, 'crit': crit, 'qc': qc, 'quant': quant, 'descr': descr, 'fdata': fdata}
    #         return od
    #
    #     # signal / noise
    #     scans = np.unique(x_sid)
    #     scan_id = np.concatenate([np.where((self.Xf[4] == x) & (self.Xf[5] == 1))[0] for x in scans])
    #     noiI = np.median(self.Xf[2, scan_id])
    #     sino = bcor_max / noiI
    #
    #     if sino < self.qc_par['sino']:
    #         crit = 'sino: s/n below qc threshold'
    #         qc = {'sId_gap': sid_gap, 'non_neg': nneg, 'sino': sino, 'ppm': ppm}
    #         quant = {}
    #         descr = {'ndp': iLen, 'st_span': np.round(rt_max - rt_min, 1), \
    #                  'mz_min': mz_min, 'mz_max': mz_max, \
    #                  'rt_min': rt_min, 'rt_max': rt_max, }
    #         fdata = {'st': x, 'st_sm': xsm, 'I_sm_bline': ybcor * ysm_max, 'I_smooth': ysm * ysm_max,
    #                  'I_raw': y * y_max,
    #                  'I_bl': y_bl * ysm_max, }
    #         od = {'flev': 2, 'crit': crit, 'qc': qc, 'quant': quant, 'descr': descr, 'fdata': fdata}
    #         return od
    #
    #     # intensity variation level
    #     idx_maxI = np.argmax(ybcor)
    #     sdxdy = (np.sum(np.diff(ybcor[:idx_maxI]) < (-0.05)) + np.sum(np.diff(ybcor[(idx_maxI + 1):]) > 0.05)) / (
    #             len(ybcor) - 3)
    #
    #     if sdxdy > self.qc_par['raggedness']:
    #         crit = 'raggedness: intensity variation level above qc threshold'
    #         qc = {'sId_gap': sid_gap, 'non_neg': nneg, 'sino': sino, 'raggedness': sdxdy, 'ppm': ppm}
    #         quant = {}
    #         descr = {'ndp': iLen, 'st_span': np.round(rt_max - rt_min, 1), \
    #                  'mz_min': mz_min, 'mz_max': mz_max, \
    #                  'rt_min': rt_min, 'rt_max': rt_max, }
    #         fdata = {'st': x, 'st_sm': xsm, 'I_sm_bline': ybcor * ysm_max, 'I_smooth': ysm * ysm_max,
    #                  'I_raw': y * y_max,
    #                  'I_bl': y_bl * ysm_max, }
    #         od = {'flev': 2, 'crit': crit, 'qc': qc, 'quant': quant, 'descr': descr, 'fdata': fdata}
    #         return od
    #
    #     # general feature descriptors
    #     # mz_maxI = np.max(self.Xf[2, idx])
    #     mz_maxI = mz[np.argmax(y)]
    #     rt_maxI = x[np.argmax(y)]
    #
    #     # rt_maxI = x[np.argmax(self.Xf[2, idx])]
    #     rt_mean = np.mean(xsm)
    #     mz_mean = np.mean(mz)
    #
    #     # create subset with equal endings
    #     _, icent = wImean(xsm, ybcor)
    #
    #     # tailing
    #     tail = rt_maxI / rt_mean
    #
    #     # integrals
    #     a = integrate.trapz(ybcor * bcor_max, x=xsm)
    #     a_raw = integrate.trapz(y, x)
    #     a_sm = integrate.trapz(ysm, xsm)
    #
    #     # I mind not need this here
    #     # include symmetry and monotonicity
    #     pl = find_peaks(ybcor * bcor_max, distance=10)
    #     plle = len(pl[0])
    #
    #     if plle > 0:
    #         pp = peak_prominences(ybcor * bcor_max, pl[0])
    #         if len(pp[0]) > 0:
    #             aa = str(np.round(pp[0][0]))
    #         else:
    #             aa = 0
    #     else:
    #         pl = -1
    #         aa = -1
    #     qc = {'sId_gap': sid_gap, 'non_neg': nneg, 'sino': sino, 'raggedness': sdxdy, 'tail': tail, 'ppm': ppm}
    #     quant = {'raw': a_raw, 'sm': a_sm, 'smbl': a}
    #     descr = {'ndp': iLen, 'npeaks': plle, 'pprom': aa, 'st_span': np.round(rt_max - rt_min, 1), \
    #              'mz_maxI': mz_maxI, 'rt_maxI': rt_maxI, \
    #              'mz_mean': mz_mean, 'mz_min': mz_min, 'mz_max': mz_max, \
    #              'rt_mean': rt_mean, 'rt_min': rt_min, 'rt_max': rt_max, }
    #     fdata = {'st': x, 'st_sm': xsm, 'I_sm_bline': ybcor * ysm_max, 'I_smooth': ysm * ysm_max, 'I_raw': y * y_max,
    #              'I_bl': y_bl * ysm_max, }
    #     od = {'flev': 3, 'qc': qc, 'quant': quant, 'descr': descr, 'fdata': fdata}
    #     return od

    # @background
    def _cls(self, cl_id, Xf, qc_par):
        """Cluster summary"""
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

    # @log
    def vizpp(self, selection: dict = {'mz_min': 425, 'mz_max': 440, 'rt_min': 400, 'rt_max': 440}, lev: int = 3):
        """Visualise feature (peak picked) data.

        Interactive scantime vs m/z plot with features highlighted, m/z and Intensity plot of selected features

        Args:
            selection: M/z and scantime (in sec) window
            lev: Feature level for marking features (`2` or `3`)
        """
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
        Xsub = self._window_mz_rt(self.Xf, selection, allow_none=False, return_idc=False)
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
            _vis_feature(fdict=self.feat[ii], id=ii, ax=axs[0])

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
                _vis_feature(self.feat['id:' + ids], id=ids, ax=axs[0], add=False)
                event.canvas.draw()
            if event.mouseevent.button is MouseButton.RIGHT:
                # print('right click')
                # print('id:' + str(ids))
                _vis_feature(self.feat['id:' + ids], id=ids, ax=axs[0], add=True)
                event.canvas.draw()

        cid1 = fig.canvas.mpl_connect('pick_event', p_text)
        axs[1].set_xlabel(r"$\bfScan time$, s")
        axs[1].yaxis.offsetText.set_visible(False)
        axs[1].yaxis.set_label_text(r"$\bfm/z$")

    # @log
    def _l3toDf(self):
        """Produce L3 feature table."""
        if len(self.feat_l3) > 0:
            descr = pd.DataFrame([self.feat[x]['descr'] for x in self.feat_l3])
            descr['id'] = self.feat_l3
            quant = pd.DataFrame([self.feat[x]['quant'] for x in self.feat_l3])
            qc = pd.DataFrame([self.feat[x]['qc'] for x in self.feat_l3])

            # out =  pd.concat((descr[[ 'rt_maxI', 'mz_maxI', 'rt_min', 'rt_max', 'npeaks']], qc[['ppm', 'sino']], quant['smbl']), axis=1)

            out = pd.concat((descr, quant, qc), axis=1)
            out.index = descr['id']
            self.l3Df = out.sort_values(['rtMaxI', 'mzMaxI', 'smbl'], ascending=True)
        else:
            print('No L3 feat')

    # @log
    # def l3gr(self):
    #     # L3 feature grouping
    #     self._l3toDf()
    #     A = self.l3Df.index
    #     isop = np.zeros(self.l3Df.shape[0])
    #     count_isop=0
    #     while len(A) > 0:
    #         id= self.l3Df.loc[A[0]]
    #         idc = np.where(((id.rt_maxI - self.l3Df.rt_maxI).abs() < 1) & ((id.mz_mean - self.l3Df.mz_mean) < 4.2))[0]
    #         isop[idc]=count_isop
    #         count_isop += 1
    #         A = list(set(A) - set(self.l3Df.index[idc]))
    #     self.l3Df['isot'] = isop
    #     self.l3Df = self.l3Df.sort_values('isot', ascending=True)


@typechecked
class Viz:
    """ Base class defining visualisation modalities for MS data.

    Note that this class is designed for use with `MsExp`.
    """

    def viz(self, q_noise: float = 0.89,
            selection: dict = {'mz_min': None, 'mz_max': None, 'rt_min': None, 'rt_max': None}, qcm_local: bool = True):
        """Interactive visualisation of scantime vs m/z dimension.

                Args:
                    q_noise: Quantile probability defining noise intensity threshold (`float`).
                    selection: `dict` of scantime (in seconds) and m/z window ranges
                    qcm_local: `bool` indicting if local quantile values should be calculated for intensity threshold and colouring

                ### Usage:
                The following example demonstrates the use of the vizd function:
                ```python
                  import msmate as ms
                  d = ms.MsExp.bruker(msmate)

                  s1 = {'mz_min': 100, 'mz_max': 115, 'rt_min': 60, 'rt_max': 70}
                  d.viz(q_noise=0.99, selection = s1, qcm_local=True)
                ```
        """
        Xsub = self._window_mz_rt(self.xrawd[self.ms0string], selection, allow_none=False)
        idc_below, idc_above = self._noiseT(p=q_noise, X=Xsub, local=qcm_local)
        cm = plt.cm.get_cmap('rainbow')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.text(1.25, 0, self.dpath, rotation=90, fontsize=6, transform=ax.transAxes)

        if len(idc_below) > 0:
            ax.scatter(Xsub[3, idc_below], Xsub[1, idc_below], s=0.1, c='gray', alpha=0.5)

        if len(idc_above) > 0:
            im = ax.scatter(Xsub[3, idc_above], Xsub[1, idc_above], c=Xsub[2, idc_above], s=5, cmap=cm, norm=LogNorm())
            fig.colorbar(im, ax=ax)
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
        ax2.set_xticklabels(self._tick_conv(tick_loc))
        ax2.set_xlabel(r"[min]")
        fig.show()
        return (fig, ax)

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

    def chromg(self, tmin: float = None, tmax: float = None, ctype: list = ['tic', 'bpc', 'xic'], xic_mz: float = [],
               xic_ppm: float = 10):
        df_l1 = self.dfd[self.ms0string]
        if df_l1.shape[0] == 0: raise ValueError('mslevel 1 in df does not exist or is not defined')
        xic = False
        tic = False
        bpc = False
        if not isinstance(ctype, list):
            ctype = [ctype]
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
            xx, xy = self._xic(xic_mz, xic_ppm, rt_min=tmin, rt_max=tmax)
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
        tick_loc = np.arange(np.round(tmin / 60), np.round(tmax / 60), 2, dtype=float) * 60
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xticks(tick_loc)
        ax2.set_xticklabels(self._tick_conv(tick_loc))
        ax2.set_xlabel(r"$\bfScantime (min)$")
        ax1.legend(loc='best', bbox_to_anchor=(0.5, 0.5, 0.5, 0.5))
        fig.show()
        return fig, ax1, ax2

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
    def vizd(self, q_noise: float = 0.50,
             selection: dict = {'mz_min': None, 'mz_max': None, 'rt_min': None, 'rt_max': None},
             qcm_local: bool = True):
        """Interactive visualisation of scantime vs m/z dimension, including an m/z density panel.

        Args:
            q_noise: Quantile probability defining noise intensity threshold (`float`).
            selection: `dict` of scantime (in seconds) and m/z window ranges
            qcm_local: `bool` indicting if local quantile values should be calculated for intensity threshold and colouring
        ### Usage:
        The following example demonstrates the use of the vizd function:
        ```python
          import msmate as ms
          d = ms.MsExp.bruker(msmate)

          s1 = {'mz_min': 100, 'mz_max': 115, 'rt_min': 60, 'rt_max': 70}
          d.vizd(q_noise=0.99, selection = s1, qcm_local=True)
        ```
        """

        def _on_lims_change(event_ax: plt.axis):
            x1, x2 = event_ax.get_xlim()
            y1, y2 = event_ax.get_ylim()
            selection = {'mz_min': y1, 'mz_max': y2, 'rt_min': x1, 'rt_max': x2}
            Xs = self._window_mz_rt(self.xrawd[self.ms0string], selection, allow_none=False)
            # print(np.where(Xs[2] > self.noise_thres)[0])
            #Xs = Xs[:,Xs[2] > self.noise_thres]
            axs[1].clear()
            axs[1].set_ylim([y1, y2])
            axs[1].tick_params(labelleft=False)
            axs[1].set_xticks([])
            axs[1].text(1.05, 0, self.fname, rotation=90, fontsize=4, transform=axs[1].transAxes)
            axs[1].tick_params(axis='y', direction='in')
            if Xs.shape[0] >0:
                bsMin = (y2 - y1) / 200 if ((y2 - y1) / 200) > 0.0001 else 0.0001
                bw = bsMin
                y, x = self._get_density(Xs, bw, q_noise=q_noise)
                axs[1].plot(x, y, c='black')
                axs[1].annotate(f'bw: {np.round(bw, 5)}\np: {np.round(q_noise, 3)}',
                                xy=(1, y1 + np.min([0.1, float((y2 - y1) / 100)])), fontsize='xx-small', ha='right')


        Xsub = self._window_mz_rt(self.xrawd[self.ms0string], selection, allow_none=False)
        idc_below, idc_above = self._noiseT(p=q_noise, X=Xsub, local=qcm_local)
        cm = plt.cm.get_cmap('rainbow')
        fig, axs = plt.subplots(1, 2, sharey=False, gridspec_kw={'width_ratios': [3, 1]})

        axs[1].tick_params(labelleft=False)
        axs[1].set_xticks([])
        axs[1].text(1.05, 0, self.fname, rotation=90, fontsize=4, transform=axs[1].transAxes)
        axs[1].tick_params(axis='y', direction='in')

        if len(idc_below) > 0:
            axs[0].scatter(Xsub[3, idc_below], Xsub[1, idc_below], s=0.1, c='gray', alpha=0.5)

        if len(idc_above) > 0:
            im = axs[0].scatter(Xsub[3, idc_above], Xsub[1, idc_above], c=Xsub[2, idc_above], s=5, cmap=cm,
                                norm=LogNorm())
            cbaxes = fig.add_axes([0.15, 0.77, 0.021, 0.1])
            cb = fig.colorbar(mappable=im, cax=cbaxes, orientation='vertical',
                              format=LogFormatterSciNotation(base=10, labelOnlyBase=False))
            cb.ax.tick_params(labelsize='xx-small')
            axs[0].callbacks.connect('ylim_changed', _on_lims_change)
        fig.subplots_adjust(wspace=0.05)
        fig.show()
        return (fig, axs)
    def _xic(self, mz: float, ppm: float, rt_min: float = None, rt_max: float = None):
        mz_min, mz_max = self._d_ppm(mz, ppm)

        idx_mz = np.where((self.xrawd[self.ms0string][1] >= mz_min) & (self.xrawd[self.ms0string] <= mz_max))[0]
        if len(idx_mz) < 0:
            raise ValueError('mz range not found')

        X_mz = self.xrawd[self.ms0string][..., idx_mz]

        sid = np.array(np.unique(X_mz[0]).astype(int))
        xic = np.zeros(int(np.max(self.xrawd[self.ms0string][0]) + 1))
        for i in sid:
            xic[i - 1] = np.sum(X_mz[np.where(X_mz[:, 0] == i), 2])
        stime = np.sort(np.unique(self.xrawd[self.ms0string][3]))

        if (~isinstance(rt_min, type(None)) | ~isinstance(rt_max, type(None))):
            idx_rt = np.where((stime >= rt_min) & (stime <= rt_max))[0]
            if len(idx_rt) < 0:
                raise ValueError('rt range not found')
            stime = stime[idx_rt]
            xic = xic[idx_rt]

        return (stime, xic)
    @staticmethod
    def _tick_conv(X):
        """Conversion time dimension from seconds to minutes."""
        V = X / 60
        return ["%.2f" % z for z in V]

class IsoPat:

    @staticmethod
    def _fwhmBound(intens, st, rtDelta=1 - 0.4):
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
        self._l3toDf()
        au = self.l3Df
        rtd = 0.6  # fwhm proportion
        fset = set(au.index.values)
        fgr = []
        c = 0
        for i in fset:
            intens = self.feat[i]['fdata']['I_sm_bline']
            st = self.feat[i]['fdata']['st']
            out = self._fwhmBound(intens, st, rtDelta=rtd)

            if out is None:
                continue

            sub = au[(au['rtMaxI'] < out[0]) & (au['rtMaxI'] > out[1])]
            if sub.shape[0] <= 1:
                continue

            sidx = sub.index[np.argmax(sub['smbl'].values)]
            intens1 = self.feat[sidx]['fdata']['I_sm_bline']
            st1 = self.feat[sidx]['fdata']['st']
            out1 = self._fwhmBound(intens1, st1, rtDelta=rtd)
            sub1 = au[(au['rtMaxI'] < out1[0]) & (au['rtMaxI'] > out1[1])].copy()
            sub1['fgr'] = c
            sub1 = sub1.sort_values('mzMaxI')
            # tt=(sub1, mzD_m1(sub1))
            ip2=(self.mzD_m1(sub1))

            if ip2.shape[0]>1:
                print(ip2)
                print('___')


            c += 1

            if any(sub1['smbl'] > 1e6):
                fgr.append(sub1)
            fset = fset - set(sub1.index.values)

        return pd.concat(fgr)  # candidate isotopologues

    @staticmethod
    def mzD_m1(df, mz_tol=0.005, a_lb=0.05):
        df = df.sort_values('smbl', ascending=False)

        m0_mz = df['mzMaxI'].iloc[0]
        m0_a = df['smbl'].iloc[0]

        dmz = df['mzMaxI'].iloc[1:] - m0_mz
        imz = (dmz > (1 - mz_tol)) & (dmz < (1 + mz_tol))
        ia = df['smbl'].iloc[1:] < (m0_a * a_lb)

        return pd.concat((df.iloc[[0]], df.iloc[(np.where(imz & ia)[0]+1)]), axis=0)



# filter for mzdiffs
# filter for intenschanges


@typechecked
class MsExp(MSstat, Fdet, Viz, IsoPat):
    """Class representing single experiment XC-MS data.

    Methods include read-in, feature detection and visualisation of chromatogram, ms scan and chromatographic vs ms dimension
    """
    @classmethod
    def mzml(cls, fpath:str, mslev:str='1'):
        # cls.__name__ = 'mzml from file import'
        da = ReadM(fpath, mslev)
        return cls(dpath=da.fpath, fname=da.fpath, mslevel=da.mslevel, ms0string=da.ms0string, ms1string=da.ms1string,
                   xrawd=da.xrawd, dfd=da.dfd, summary=False)

    @classmethod
    def bruker(cls, dpath: str, convert: bool = True, docker: dict = {'repo': 'convo:v1', 'mslevel': 0, 'smode': [0, 1, 2, 3, 4, 5], 'amode': 2, 'seg': [0, 1, 2, 3], }):
        # cls.__name__ = 'mzml from bruker directory'
        da = ReadB(dpath, convert, docker)
        return cls(dpath = da.dpath, fname=da.fname, mslevel=da.mslevel, ms0string=da.ms0string, ms1string=da.ms1string, xrawd=da.xrawd, dfd=da.dfd, summary=da.summary)

    @classmethod
    def rM(cls, da: ReadM):
        # cls.__name__ = 'data import from msmate ReadM'
        return cls(dpath=da.fpath, fname=da.fpath, mslevel=da.mslevel, ms0string=da.ms0string, ms1string=da.ms1string,
                   xrawd=da.xrawd, dfd=da.dfd, summary=False)

    @classmethod
    def rB(cls, da: ReadB):
        # cls.__name__ = 'data import from msmate ReadB'
        return cls(dpath=da.dpath, fname=da.fname, mslevel=da.mslevel, ms0string=da.ms0string, ms1string=da.ms1string,
                   xrawd=da.xrawd, dfd=da.dfd, summary=da.summary)

    # @logIA
    def __init__(self, dpath:str, fname:str, mslevel:str, ms0string:str, ms1string:Union[str, None], xrawd:dict, dfd:dict, summary:bool):
        self.mslevel = mslevel
        self.dpath = dpath
        self.fname = fname

        self.ms0string = ms0string
        self.ms1string = ms1string
        self.xrawd = xrawd # usage: self.xrawd[self.ms0string]
        self.dfd = dfd # usage: self.dfd[scantype], scantype eg., '0_2_2_5_20.0'
        # self.df
        self.summary = summary


# class list_exp:
#     def __init__(self, path, ftype='mzml', nmax=None, print_summary=True):
#
#         import pandas as pd
#         import numpy as np
#
#         self.ftype = ftype
#         self.path = path
#         self.files = []
#         self.nmax=nmax
#
#         import glob, os
#         self.path=os.path.join(path, '')
#
#         c = 1
#         for file in glob.glob(self.path+"*."+ftype):
#             self.files.append(file)
#             if c == nmax: break
#             c += 1
#
#
#         # cmd = 'find ' + self.path + ' -type f -iname *' + ftype
#         # sp = subprocess.getoutput(cmd)
#         # self.files = sp.split('\n')
#         self.nfiles = len(self.files)
#
#         # check if bruker fits and qc's are included
#         fsize = list()
#         mtime = list()
#         # check if procs exists
#         for i in range(self.nfiles):
#             fname = self.files[i]
#             inf = os.stat(fname)
#             try:
#                 inf
#             except NameError:
#                 mtime.append(None)
#                 fsize.append(None)
#                 continue
#
#             mtime.append(int(inf.st_mtime))
#             fsize.append(int(inf.st_size))
#
#         self.df = pd.DataFrame({'exp': self.files, 'fsize_mb': fsize, 'mtime': mtime})
#         self.df['fsize_mb'] = fsize
#         self.df['fsize_mb'] = self.df['fsize_mb'].values / 1000 ** 2
#         self.df['mtime'] = mtime
#         mtime_max, mtime_min = (np.argmax(self.df['mtime'].values), np.argmin(self.df['mtime'].values))
#         fsize_mb_max, fsize_mb_min = (np.argmax(self.df['fsize_mb'].values), np.argmin(self.df['fsize_mb'].values))
#         self.df['mtime'] = pd.to_datetime(self.df['mtime'], unit='s').dt.floor('T')
#
#         self.df.exp = [os.path.relpath(self.df.exp.values[x], path) for x in range(len(self.df.exp.values))]
#         self.files =  self.df.exp.values
#         # os.path.relpath(self.df.exp.values[0], path)
#
#         self.df.index = ['s' + str(i) for i in range(self.df.shape[0])]
#
#         self.n_exp = (self.df.shape[0])
#
#         if print_summary:
#             print('---')
#             print(f'Number of mzml files: {self.n_exp}')
#             print(f'Files created from { self.df.mtime.iloc[mtime_min]} to { self.df.mtime.iloc[mtime_max]}')
#             print(f'Files size ranging from {round(self.df.fsize_mb.iloc[fsize_mb_min],1)} MB to {round(self.df.fsize_mb.iloc[fsize_mb_max], 1)} MB')
#             print(f'For detailed experiment list see obj.df')
#             print('---')
#
#
# class msExp:
#     from msmate._peak_picking import est_intensities, cl_summary, feat_summary1, peak_picking, vis_feature_pp
#     #from ._reading import read_mzml, read_bin, exp_meta, node_attr_recurse, rec_attrib, extract_mass_spectrum_lev1
#     from ._isoPattern import element, element_list, calc_mzIsotopes
#
#     def __init__(self, fpath, mslev='1', print_summary=False):
#         self.fpath = fpath
#         self.read_exp(mslevel=mslev, print_summary=print_summary)
#
#     # def peak(self, st_len_min=10, plot=True, mz_bound=0.001, rt_bound=3, qc_par={'st_len_min': 5, 'raggedness': 0.15, 'non_neg': 0.8, 'sId_gap': 0.15, 'sino': 2, 'ppm': 15}):
#     #     self.st_len_min=st_len_min
#     #     self.qc_par = qc_par
#     #     self.peak_picking(self.X2, self.st_len_min,)
#
#     def read_exp(self, mslevel='1', print_summary=True):
#         import numpy as np
#
#         #if fidx > (len(self.files)-1): print('Index too high (max value is ' + str(self.nfiles-1) + ').')
#         self.flag=str(mslevel)
#
#         out = self.read_mzml(flag=self.flag, print_summary=print_summary)
#         print('file: ' + self.fpath + '-->' + out[0])
#
#         if 'srm' in out[0]:
#             self.dstype, self.df, self.scantime, self.I, self.summary = out
#
#         if 'untargeted' in out[0]:
#             self.dstype, self.df, self.scanid, self.mz, self.I, self.scantime, self.summary = out
#             # out = self.read_mzml(path=path, sidx=self.fidx, flag=self.flag, print_summary=print_summary)
#             self.nd = len(self.I)
#             self.Xraw = np.array([self.scanid, self.mz, self.I, self.scantime]).T
#         self.stime_conv()
#
#     def d_ppm(self, mz, ppm):
#         d = (ppm * mz)/ 1e6
#         return mz-(d/2), mz+(d/2)
#
#     def xic(self, mz, ppm, rt_min=None, rt_max=None):
#         import numpy as np
#         mz_min, mz_max = self.d_ppm(mz, ppm)
#
#         idx_mz = np.where((self.Xraw[:, 1] >= mz_min) & (self.Xraw[:, 1] <= mz_max))[0]
#         if len(idx_mz) < 0:
#             raise ValueError('mz range not found')
#
#         X_mz = self.Xraw[idx_mz, :]
#
#         sid = np.array(np.unique(X_mz[:, 0]).astype(int))
#         xic = np.zeros(int(np.max(self.Xraw[:, 0])+1))
#         for i in sid:
#             xic[i-1] = np.sum(X_mz[np.where(X_mz[:,0] == i), 2])
#         stime = np.sort(np.unique(self.Xraw[:,3]))
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
#
#     @staticmethod
#     def window_mz_rt(Xr, selection={'mz_min': None, 'mz_max': None, 'rt_min': None, 'rt_max': None},
#                      allow_none=False, return_idc=False):
#         # make selection on Xr based on rt and mz range
#         # allow_none: true if None alowed in selection (then full range is used)
#
#         import numpy as np
#         part_none = any([x == None for x in selection.values()])
#         all_fill = any([x != None for x in selection.values()])
#
#         if not allow_none:
#             if not all_fill:
#                 raise ValueError('Define plot boundaries with selection argument.')
#         if part_none:
#             if selection['mz_min'] is not None:
#                 t1 = Xr[..., 1] >= selection['mz_min']
#             else:
#                 t1 = np.ones(Xr.shape[0], dtype=bool)
#
#             if selection['mz_max'] is not None:
#                 t2 = Xr[..., 1] <= selection['mz_max']
#             else:
#                 t2 = np.ones(Xr.shape[0], dtype=bool)
#
#             if selection['rt_min'] is not None:
#                 t3 = Xr[..., 3] > selection['rt_min']
#             else:
#                 t3 = np.ones(Xr.shape[0], dtype=bool)
#
#             if selection['rt_max'] is not None:
#                 t4 = Xr[..., 3] < selection['rt_max']
#             else:
#                 t4 = np.ones(Xr.shape[0], dtype=bool)
#         else:
#             t1 = Xr[..., 1] >= selection['mz_min']
#             t2 = Xr[..., 1] <= selection['mz_max']
#             t3 = Xr[..., 3] > selection['rt_min']
#             t4 = Xr[..., 3] < selection['rt_max']
#
#         idx = np.where(t1 & t2 & t3 & t4)[0]
#
#         if return_idc:
#             return idx
#         else:
#             return Xr[idx, ...]
#
#     def vis_spectrum(self, q_noise=0.89, selection={'mz_min': None, 'mz_max': None, 'rt_min': None, 'rt_max': None}):
#         # visualise part of the spectrum, before peak picking
#
#         import matplotlib.pyplot as plt
#         import numpy as np
#
#         # select Xraw
#         Xsub = self.window_mz_rt(self.Xraw, selection, allow_none=False)
#
#         # filter noise points
#         noise_thres = np.quantile(self.Xraw[..., 2], q=q_noise)
#         idc_above = np.where(Xsub[:, 2] > noise_thres)[0]
#         idc_below = np.where(Xsub[:, 2] <= noise_thres)[0]
#
#         cm = plt.cm.get_cmap('rainbow')
#         # fig, ax = plt.subplots()
#         fig = plt.figure()
#         ax = fig.add_subplot(111)
#         ax.text(1.25, 0, self.df.fname[0], rotation=90, fontsize=6, transform=ax.transAxes)
#
#         ax.scatter(Xsub[idc_below, 3], Xsub[idc_below, 1], s=0.1, c='gray', alpha=0.5)
#         im = ax.scatter(Xsub[idc_above, 3], Xsub[idc_above, 1], c=np.log(Xsub[idc_above, 2]), s=5, cmap=cm)
#         fig.canvas.draw()
#
#         ax.set_xlabel(r"$\bfScan time$, s")
#         ax2 = ax.twiny()
#
#         ax.yaxis.offsetText.set_visible(False)
#         ax.yaxis.set_label_text(r"$\bfm/z$")
#
#         def tick_conv(X):
#             V = X / 60
#             return ["%.2f" % z for z in V]
#
#         tmax = np.max(Xsub[:, 3])
#         tmin = np.min(Xsub[:, 3])
#
#         tick_loc = np.linspace(tmin / 60, tmax / 60, 5) * 60
#         # tick_loc = np.arange(np.round((tmax - tmin) / 60, 0), 0.1)*60
#         ax2.set_xlim(ax.get_xlim())
#         ax2.set_xticks(tick_loc)
#         ax2.set_xticklabels(tick_conv(tick_loc))
#         ax2.set_xlabel(r"min")
#         fig.colorbar(im, ax=ax)
#         fig.show()
#
#         return (fig, ax)
#
#     def vis_features(self, selection={'mz_min': 600, 'mz_max': 660, 'rt_min': 600, 'rt_max': 900}, rt_bound=0.51,
#                      mz_bound=0.001):
#         # visualise mz/rt plot after peak picking
#         # basically like vis spectrum but including rectangles and all points but those in fgr feature(s) are s=0.1 and c='grey
#         import matplotlib.pyplot as plt
#         from matplotlib.patches import Rectangle
#         import numpy as np
#
#         # which feature - define plot axis boundaries
#         Xsub = self.window_mz_rt(self.Xf, selection, allow_none=False, return_idc=False)
#
#         labs = np.unique(Xsub[:,5])
#         # define L2 clusters
#         cl_l2 = self.fs[np.isin(self.fs.cl_id.values, labs)]
#
#         # define points exceeding noise and in L2 feature list
#         cl_n_above = np.where((Xsub[:, 4] == 0) & (np.isin(Xsub[..., 5], cl_l2.cl_id.values)))[0]
#         cl_n_below = np.where((Xsub[:, 4] == 1) | (~np.isin(Xsub[..., 5], cl_l2.cl_id.values)))[0]
#         assert ((len(cl_n_above) + len(cl_n_below)) == Xsub.shape[0])
#
#         cm = plt.cm.get_cmap('gist_ncar')
#         # fig, ax = plt.subplots()
#         fig = plt.figure()
#         ax = fig.add_subplot(111)
#         ax.text(1.25, 0, self.df.fname[0], rotation=90, fontsize=6, transform=ax.transAxes)
#         ax.scatter(Xsub[cl_n_below, 3], Xsub[cl_n_below, 1], s=0.1, c='gray', alpha=0.1)
#         for i in range(cl_l2.shape[0]):
#             f1 = cl_l2.iloc[i]
#             ax.add_patch(Rectangle(((f1['rt_min'] - rt_bound), f1['mz_min'] - mz_bound), \
#                                    ((f1['rt_max']) - (f1['rt_min'])) + rt_bound * 2, \
#                                    (f1['mz_max'] - f1['mz_min']) + mz_bound * 2, fc='gray', alpha=0.5, \
#                                    linestyle="dotted"))
#
#         im = ax.scatter(Xsub[cl_n_above, 3], Xsub[cl_n_above, 1], c=np.log(Xsub[cl_n_above, 2]), s=5, cmap=cm)
#         fig.canvas.draw()
#         ax.set_xlabel(r"$\bfScan time$, s")
#         ax2 = ax.twiny()
#
#         ax.yaxis.offsetText.set_visible(False)
#         # ax1.yaxis.set_label_text("original label" + " " + offset)
#         ax.yaxis.set_label_text(r"$\bfm/z$" )
#
#         def tick_conv(X):
#             V = X / 60
#             return ["%.2f" % z for z in V]
#
#         tmax=np.max(Xsub[:,3])
#         tmin=np.min(Xsub[:,3])
#
#         tick_loc = np.linspace(tmin/60, tmax/60, 5) * 60
#         # tick_loc = np.arange(np.round((tmax - tmin) / 60, 0), 0.1)*60
#         ax2.set_xlim(ax.get_xlim())
#         ax2.set_xticks(tick_loc)
#         ax2.set_xticklabels(tick_conv(tick_loc))
#         ax2.set_xlabel(r"min")
#         fig.colorbar(im, ax=ax)
#
#         return (fig, ax)
#
#     def stime_conv(self):
#         if 'minute' in self.df:
#             self.df['scan_start_time']=self.df['scan_start_time'].astype(float)*60
#         else:
#             self.df['scan_start_time'] = self.df['scan_start_time'].astype(float)
#
#     def find_isotopes(self):
#         import numpy as np
#         import pandas as pd
#         # deconvolution based on isotopic mass differences (1 m/z)
#         # check which features overlap in rt dimension
#
#         rt_delta = 0.1
#
#         descr = pd.DataFrame([self.feat[x]['descr'] for x in self.feat_l3])
#         qc = pd.DataFrame([self.feat[x]['qc'] for x in self.feat_l3])
#         quant = pd.DataFrame([self.feat[x]['quant'] for x in self.feat_l3])
#
#
#         fs = pd.concat([descr, qc, quant], axis=1)
#         fs.index = self.feat_l3
#
#         fs['gr'] = np.ones(fs.shape[0]) * -1
#         # test.sort_values(by=['sino'], ascending=False).id[0:30]
#
#         S = self.feat_l3.copy()
#         burn = []
#         fgroups = {}
#         c=0
#         igrp = fs.shape[1]-1
#         #for j in range(len(S)):
#         while len(S) > 0:
#             i = S[0]
#             idx = np.where((fs['rt_maxI'].values > (fs.loc[i]['rt_maxI'] - rt_delta)) & (fs['rt_maxI'].values < (fs.loc[i]['rt_maxI'] + rt_delta)) & ~fs.index.isin(burn))[0]
#             if len(idx) > 1:
#                 fs.iloc[idx, igrp]=c
#                 c += 1
#                 ids=fs.iloc[idx].index.values.tolist()
#                 burn = burn + ids
#                 [S.remove(x) for x in ids]
#                 ifound = {'seed': i, \
#                           'ridx': ids,
#                           'quant': fs.iloc[idx].raw.values,
#                           'rt_imax': fs.iloc[idx].rt_maxI.values,
#                           'mz_diff': fs.iloc[idx].mz_mean}
#                 fgroups.update({'fg'+ str(c) : ifound})
#             else:
#                 S.remove(i)
#
#         print(f'{c-1} feature groups detected')
#         self.fgroups = fgroups
#         self.fs = fs
#
#     def vis_feature_gr(self, fgr, b_mz=4, b_rt=2):
#         # l2 and l1 features mz/rt plot after peak picking
#         # label feature with id
#         # two panel plot: bottom m/z over st, top: st vs intentsity upon click on feature
#         # basically like vis spectrum but including rectangles and all points but those in fgr feature(s) are s=0.1 and c='grey
#         import matplotlib.pyplot as plt
#         from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#         from matplotlib.backend_bases import MouseButton
#         from matplotlib.patches import Rectangle
#         import numpy as np
#         import pandas as pd
#         from itertools import compress
#
#         # which feature - define plot axis boundaries
#         mz_min = self.fs.mz_min.loc[fgr['ridx']].min() - b_mz / 2
#         mz_max = self.fs.mz_max.loc[fgr['ridx']].max() + b_mz / 2
#
#         rt_min = self.fs.rt_min.loc[fgr['ridx']].min() - b_rt / 2
#         rt_max = self.fs.rt_max.loc[fgr['ridx']].max() + b_rt / 2
#
#         selection = {'mz_min': mz_min, 'mz_max': mz_max, 'rt_min': rt_min, 'rt_max': rt_max}
#
#         def vis_feature(fdict, id, ax=None, add=False):
#             import matplotlib.pyplot as plt
#
#             # idx = np.where(self.Xf[:, 6] == fdict['id'])[0]
#             if isinstance(ax, type(None)):
#                 fig = plt.figure()
#                 ax = fig.add_subplot(111)
#
#             ax.plot(fdict['fdata']['st'], fdict['fdata']['I_raw'], label='raw', linewidth=0.5, color='black', zorder=0)
#
#             if fdict['flev'] == 3:
#                 ax.plot(fdict['fdata']['st_sm'], fdict['fdata']['I_smooth'], label='smoothed', linewidth=0.5, color='cyan', zorder=0)
#                 ax.plot(fdict['fdata']['st_sm'], fdict['fdata']['I_bl'], label='baseline', linewidth=0.5, color='black', ls='dashed', zorder=0)
#                 ax.plot(fdict['fdata']['st_sm'], fdict['fdata']['I_sm_bline'], label='smoothed bl-cor', color='orange', zorder=0)
#             if not add:
#                 ax.set_title(id + f',  flev: {fdict["flev"]}')
#                 ax.legend()
#             else:
#                 old= axs[0].get_title()
#                 ax.set_title(old + '\n' +  id + f',  flev: {fdict["flev"]}')
#             ax.set_ylabel(r"$\bfCount$")
#             ax.set_xlabel(r"$\bfScan time$, s")
#
#             data = np.array([list(fdict['qc'].keys()), np.round(list(fdict['qc'].values()), 2)]).T
#             column_labels = ["descr", "value"]
#             ax.table( cellText=data, colLabels=column_labels, loc='lower right', colWidths = [0.1]*3)
#
#             return ax
#
#         # Xf has 7 columns: 0 - scanId, 1 - mz, 2 - int, 3 - st, 4 - noiseBool, 5 - st_adj, 6 - clMem
#
#         # which feature - define plot axis boundaries
#         Xsub = self.window_mz_rt(self.Xf, selection, allow_none=False, return_idc=False)
#         fid_window = np.unique(Xsub[:,6])
#         l3_ffeat = list(compress(self.feat_l3, np.in1d([float(i.split(':')[1]) for i in self.feat_l3], fid_window)))
#
#         cm = plt.cm.get_cmap('gist_ncar')
#         fig, axs = plt.subplots(2,1, gridspec_kw={'height_ratios': [1, 2]}, figsize=(8,10), constrained_layout=False)
#         fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
#         axs[0].set_ylabel(r"$\bfCount$")
#         axs[0].set_xlabel(r"$\bfScan time$, s")
#
#         vis_feature(self.feat[self.feat_l2[0]], id=self.feat_l2[0], ax=axs[0])
#         axs[1].set_facecolor('#EBFFFF')
#         axs[1].text(1.03, 0, self.df.fname[0], rotation=90, fontsize=6, transform=axs[1].transAxes)
#         axs[1].scatter(Xsub[Xsub[:,4]==0, 3], Xsub[Xsub[:,4]==0, 1], s=0.1, c='gray', alpha=0.5)
#
#         for i in l3_ffeat:
#             fid = float(i.split(':')[1])
#             # Xsub[Xsub[:,6] == fid, :]
#             f1 = self.feat[i]
#             a_col = 'grey'
#             fsize = 6
#
#             if any([i==x for x in fgr['ridx']]):
#                 a_col = 'green'
#                 fsize = 10
#
#             mz=Xsub[Xsub[:, 6] == fid, 1]
#             if len(mz) == len(f1['fdata']['st']):
#                 im = axs[1].scatter(f1['fdata']['st'], mz,
#                                     c=np.log(f1['fdata']['I_raw']), s=5, cmap=cm)
#                 axs[1].add_patch(Rectangle(((f1['descr']['rt_min']), f1['descr']['mz_min']), \
#                                            ((f1['descr']['rt_max']) - (f1['descr']['rt_min'])), \
#                                            (f1['descr']['mz_max'] - f1['descr']['mz_min']), fc='#C2D0D6', \
#                                            linestyle="solid", color='red', linewidth=2, zorder=0, in_layout=True,
#                                            picker=False))
#                 axs[1].annotate(i.split(':')[1], (f1['descr']['rt_max'], f1['descr']['mz_max']),
#                                 bbox=dict(facecolor=a_col, alpha=0.3, boxstyle='circle', edgecolor='white',
#                                           in_layout=False),
#                                 wrap=False, picker=True, fontsize=fsize)
#
#         # coords = pd.DataFrame(coords)
#         cbaxes = inset_axes(axs[1], width="30%", height="5%", loc=3)
#         plt.colorbar(im, cax=cbaxes, ticks=[0., 1], orientation='horizontal')
#
#         def p_text(event):
#             print(event.artist)
#             ids = str(event.artist.get_text())
#             if event.mouseevent.button is MouseButton.LEFT:
#                 print('left click')
#                 print('id:' + str(ids))
#                 axs[0].clear()
#                 axs[0].set_title('')
#                 vis_feature(self.feat['id:' + ids], id=ids, ax=axs[0], add=False)
#             if event.mouseevent.button is MouseButton.RIGHT:
#                 print('right click')
#                 print('id:' + str(ids))
#                 vis_feature(self.feat['id:' + ids], id=ids, ax=axs[0], add=True)
#
#         cid1 = fig.canvas.mpl_connect('pick_event', p_text)
#         axs[1].set_xlabel(r"$\bfScan time$, s")
#         axs[1].yaxis.offsetText.set_visible(False)
#         axs[1].yaxis.set_label_text(r"$\bfm/z$")
#
#     def vis_fgroup(self, fgr, b_mz=2, b_rt=2, rt_bound=0.51, mz_bound=0.2):
#         # visualise mz/rt plot after peak picking and isotope detection
#         # basically like vis spectrum but including rectangles and all points but those in fgr feature(s) are s=0.1 and c='grey
#         import matplotlib.pyplot as plt
#         from matplotlib.patches import Rectangle
#         import numpy as np
#
#         # which feature - define plot axis boundaries
#         mz_min = self.fs.mz_min.loc[fgr['ridx']].min() - b_mz / 2
#         mz_max = self.fs.mz_max.loc[fgr['ridx']].max() + b_mz / 2
#
#         rt_min = self.fs.rt_min.loc[fgr['ridx']].min() - b_rt / 2
#         rt_max = self.fs.rt_max.loc[fgr['ridx']].max() + b_rt / 2
#
#         Xsub = self.window_mz_rt(self.Xf,
#                                  selection={'mz_min': mz_min, 'mz_max': mz_max, 'rt_min': rt_min, 'rt_max': rt_max},
#                                  allow_none=False, return_idc=False)
#
#         # define L2 clusters
#         cl_l2 = self.fs.loc[fgr['ridx']]
#
#         # define points exceeding noise and in L2 feature list
#         # cl_n_above = np.where((Xsub[:, 4] == 0) & (np.isin(Xsub[..., 5], cl_l2.cl_id.values)))[0]
#         # cl_n_below = np.where((Xsub[:, 4] == 1) | (~np.isin(Xsub[..., 5], cl_l2.cl_id.values)))[0]
#         noi = np.where(Xsub[:, 4] == 1)[0]
#         # assert ((len(cl_n_above) + len(cl_n_below)) == Xsub.shape[0])
#
#         cm = plt.cm.get_cmap('gist_ncar')
#         # fig, ax = plt.subplots()
#         fig = plt.figure()
#         ax = fig.add_subplot(111)
#         ax.text(1.25, 0, self.df.fname[0], rotation=90, fontsize=6, transform=ax.transAxes)
#         ax.scatter(Xsub[noi, 3], Xsub[noi, 1], s=0.5, c='gray', alpha=0.7)
#         for i in range(cl_l2.shape[0]):
#             f1 = cl_l2.iloc[i]
#             ax.add_patch(Rectangle(((f1['rt_min'] - rt_bound), f1['mz_min'] - mz_bound), \
#                                    ((f1['rt_max']) - (f1['rt_min'])) + rt_bound * 2, \
#                                    (f1['mz_max'] - f1['mz_min']) + mz_bound * 2, fc='gray', alpha=0.5, \
#                                    linestyle="dotted"))
#
#         im = ax.scatter(Xsub[cl_n_above, 3], Xsub[cl_n_above, 1], c=np.log(Xsub[cl_n_above, 2]), s=5, cmap=cm)
#         fig.canvas.draw()
#         ax.set_xlabel(r"$\bfScan time$, s")
#         ax2 = ax.twiny()
#
#         ax.yaxis.offsetText.set_visible(False)
#         # ax1.yaxis.set_label_text("original label" + " " + offset)
#         ax.yaxis.set_label_text(r"$\bfm/z$")
#
#         def tick_conv(X):
#             V = X / 60
#             return ["%.2f" % z for z in V]
#
#         tmax = np.max(Xsub[:, 3])
#         tmin = np.min(Xsub[:, 3])
#
#         tick_loc = np.linspace(tmin / 60, tmax / 60, 5) * 60
#         # tick_loc = np.arange(np.round((tmax - tmin) / 60, 0), 0.1)*60
#         ax2.set_xlim(ax.get_xlim())
#         ax2.set_xticks(tick_loc)
#         ax2.set_xticklabels(tick_conv(tick_loc))
#         ax2.set_xlabel(r"min")
#         fig.colorbar(im, ax=ax)
#
#         return (fig, ax)
#
#     def vis_ipattern(self, fgr, b_mz=10, b_rt=10, rt_bound=0.51, mz_bound=0.2):
#         # visualise mz/rt plot after peak picking and isotope detection
#         # basically like vis spectrum but including rectangles and all points but those in fgr feature(s) are s=0.1 and c='grey
#         import matplotlib.pyplot as plt
#         from matplotlib.patches import Rectangle
#         import numpy as np
#
#         # which feature - define plot axis boundaries
#         mz_min = self.fs.mz_min.iloc[fgr['ridx']].min() - b_mz / 2
#         mz_max = self.fs.mz_max.iloc[fgr['ridx']].max() + b_mz / 2
#
#         rt_min = self.fs.rt_min.iloc[fgr['ridx']].min() - b_rt / 2
#         rt_max = self.fs.rt_max.iloc[fgr['ridx']].max() + b_rt / 2
#
#         Xsub = self.window_mz_rt(self.Xf,
#                                  selection={'mz_min': mz_min, 'mz_max': mz_max, 'rt_min': rt_min, 'rt_max': rt_max},
#                                  allow_none=False, return_idc=False)
#
#         # define L2 clusters
#         cl_l2 = self.fs.iloc[fgr['ridx']]
#
#         # define points exceeding noise and in L2 feature list
#         cl_n_above = np.where((Xsub[:, 4] == 0) & (np.isin(Xsub[..., 5], cl_l2.cl_id.values)))[0]
#         cl_n_below = np.where((Xsub[:, 4] == 1) | (~np.isin(Xsub[..., 5], cl_l2.cl_id.values)))[0]
#         assert ((len(cl_n_above) + len(cl_n_below)) == Xsub.shape[0])
#
#         cm = plt.cm.get_cmap('gist_ncar')
#         # fig, ax = plt.subplots()
#         fig = plt.figure()
#         ax = fig.add_subplot(111)
#         ax.text(1.25, 0, self.df.fname[0], rotation=90, fontsize=6, transform=ax.transAxes)
#         ax.scatter(Xsub[cl_n_below, 3], Xsub[cl_n_below, 1], s=0.5, c='gray', alpha=0.7)
#         for i in range(cl_l2.shape[0]):
#             f1 = cl_l2.iloc[i]
#             ax.add_patch(Rectangle(((f1['rt_min'] - rt_bound), f1['mz_min'] - mz_bound), \
#                                    ((f1['rt_max']) - (f1['rt_min'])) + rt_bound * 2, \
#                                    (f1['mz_max'] - f1['mz_min']) + mz_bound * 2, fc='gray', alpha=0.5, \
#                                    linestyle="dotted"))
#
#         im = ax.scatter(Xsub[cl_n_above, 3], Xsub[cl_n_above, 1], c=np.log(Xsub[cl_n_above, 2]), s=5, cmap=cm)
#         fig.canvas.draw()
#         ax.set_xlabel(r"$\bfScan time$, s")
#         ax2 = ax.twiny()
#
#         ax.yaxis.offsetText.set_visible(False)
#         # ax1.yaxis.set_label_text("original label" + " " + offset)
#         ax.yaxis.set_label_text(r"$\bfm/z$")
#
#         def tick_conv(X):
#             V = X / 60
#             return ["%.2f" % z for z in V]
#
#         tmax = np.max(Xsub[:, 3])
#         tmin = np.min(Xsub[:, 3])
#
#         tick_loc = np.linspace(tmin / 60, tmax / 60, 5) * 60
#         # tick_loc = np.arange(np.round((tmax - tmin) / 60, 0), 0.1)*60
#         ax2.set_xlim(ax.get_xlim())
#         ax2.set_xticks(tick_loc)
#         ax2.set_xticklabels(tick_conv(tick_loc))
#         ax2.set_xlabel(r"min")
#         fig.colorbar(im, ax=ax)
#
#         return (fig, ax)
#
#     def plt_chromatogram(self, tmin=None, tmax=None, ctype=['tic', 'bpc', 'xic'], xic_mz=[], xic_ppm=10):
#         import numpy as np
#         import matplotlib.pyplot as plt
#
#         df_l1 = self.df.loc[self.df.ms_level == '1']
#         df_l2 = self.df.loc[self.df.ms_level == '2']
#         if df_l1.shape[0] == 0: raise ValueError('mslevel 1 in df does not exist or is not defined')
#
#         xic = False
#         tic = False
#         bpc = False
#
#         if not isinstance(ctype, list):
#             ctype=[ctype]
#         if any([x in 'xic' for x in ctype]):
#             xic = True
#         if any([x in 'bpc' for x in ctype]):
#             bpc = True
#         if any([x in 'tic' for x in ctype]):
#             tic = True
#
#         if ((xic == False) & (tic == False) & (bpc == False)):
#             raise ValueError('ctype error: define chromatogram type(s) as tic, bpc or xic (string or list of strings).')
#
#         if isinstance(tmin, type(None)):
#             tmin = df_l1.scan_start_time.min()
#         if isinstance(tmax, type(None)):
#             tmax = df_l1.scan_start_time.max()
#
#         # if isinstance(ax, type(None)):
#         #
#         # else: ax1=ax
#         fig = plt.figure()
#         ax1 = fig.add_subplot(111)
#
#         if xic:
#             xx, xy = self.xic(xic_mz, xic_ppm, rt_min=tmin, rt_max=tmax)
#             ax1.plot(xx, xy.astype(float), label='XIC ' + str(xic_mz) + 'm/z')
#         if tic:
#             ax1.plot(df_l1.scan_start_time, df_l1.total_ion_current.astype(float), label='TIC')
#         if bpc:
#             ax1.plot(df_l1.scan_start_time, df_l1.base_peak_intensity.astype(float), label='BPC')
#
#         # ax1.set_title(df.fname[0], fontsize=8, y=-0.15)
#         ax1.text(1.04, 0, df_l1.fname[0], rotation=90, fontsize=6, transform=ax1.transAxes)
#
#         if df_l2.shape[0] == 0:
#             print('Exp contains no CID data')
#         else:
#             ax1.scatter(df_l2.scan_start_time, np.zeros(df_l2.shape[0]), marker=3, c='black', s=6)
#         fig.canvas.draw()
#         offset = ax1.yaxis.get_major_formatter().get_offset()
#         # offset = ax1.get_xaxis().get_offset_text()
#
#         ax1.set_xlabel(r"$\bfs$")
#         ax1.set_ylabel(r"$\bfCount$")
#         ax2 = ax1.twiny()
#
#         ax1.yaxis.offsetText.set_visible(False)
#         # ax1.yaxis.set_label_text("original label" + " " + offset)
#         if offset != '': ax1.yaxis.set_label_text(r"$\bfInt/count$ ($x 10^" + offset.replace('1e', '') + '$)')
#
#         def tick_conv(X):
#             V = X / 60
#             return ["%.2f" % z for z in V]
#
#         print(len(ax1.get_xticks()))
#         # tick_loc = np.linspace(np.round(tmin/60), np.round(tmax/60),
#         #                        (round(np.round((tmax - tmin) / 60, 0) / 2) + 1)) * 60
#         tick_loc = np.arange(np.round(tmin/60), np.round(tmax/60), 2, dtype=float) * 60
#         ax2.set_xlim(ax1.get_xlim())
#         ax2.set_xticks(tick_loc)
#         ax2.set_xticklabels(tick_conv(tick_loc))
#         ax2.set_xlabel(r"$\bfmin$")
#         # ax2.yaxis.set_label_text(r"$Int/count$ " + offset)
#
#         ax1.legend(loc='best', bbox_to_anchor=(0.5, 0.5, 0.5, 0.5))
#
#         fig.show()
#         return fig, ax1, ax2
#     def tic_chromMS(self):
#
#         import matplotlib.pyplot as plt
#         import numpy as np
#         df_l1 = self.df.loc[self.df.ms_level == '1']
#         df_l2 = self.df.loc[self.df.ms_level == '2']
#         if df_l1.shape[0] == 0: raise ValueError('mslevel 1 in df does not exist or is not defined')
#
#         fig, axs = plt.subplots(2,1)
#         axs[0].plot(df_l1.scan_start_time, df_l1.total_ion_current.astype(float), label='TIC')
#
#
#         def mass_spectrum_callback(event):
#             print(event.dblclick )
#             if ((event.inaxes in [axs[0]]) & (event.dblclick == 1)):
#                 if event.xdata <= np.max(df_l1.scan_start_time):
#                     idx_scan = np.argmin(np.abs(event.xdata - df_l1.scan_start_time))
#                     axs[0].scatter(df_l1.scan_start_time[idx_scan], df_l1.total_ion_current.astype(float)[idx_scan], c='red')
#
#                     ms_scan=float(self.df.id.iloc[idx_scan].split('scan=')[1])
#                     idc = np.where(self.Xraw[:,0]==ms_scan)[0]
#
#
#                     axs[1].clear()
#                     ymax = np.max(self.Xraw[idc, 2])
#                     y=self.Xraw[idc, 2]/ymax *100
#                     x=self.Xraw[idc, 1]
#                     axs[1].vlines(x, ymin=0, ymax=y)
#                     axs[1].set_xlim(0, 1200)
#
#                     # add labels of three max points
#                     idc_max = np.argsort(-y)[0:3]
#                     axs[1].set_ylim(0, 110)
#                     for i in range(3):
#                         axs[1].annotate(np.round(x[idc_max[i]], 3), (x[idc_max[i]], y[idc_max[i]]))
#
#             else:
#                 return
#
#         def zoom_massSp_callback(axx):
#             print(axx.get_xlim())
#             # if event.inaxes == axs[1]:
#             #     print(axs[1].get_xlim())
#
#
#         cid = fig.canvas.mpl_connect('button_press_event', mass_spectrum_callback)
#
#         axs[1].callbacks.connect('xlim_changed', zoom_massSp_callback)
#
#
#     def read_mzml(self, flag='1', print_summary=True):
#         # read in data, files index 0 (see below)
#         # flag is 1/2 for msLevel 1 or 2
#         import pandas as pd
#         import re
#         import numpy as np
#         import xml.etree.cElementTree as ET
#
#         tree = ET.parse(self.fpath)
#         root = tree.getroot()
#
#         child = children(root)
#         imzml = child.index('mzML')
#         mzml_children = children(root[imzml])
#
#         obos = root[imzml][mzml_children.index('cvList')]
#         self.obo_ids = get_obo(obos, obo_ids={})
#
#         seq = np.where(~np.isin(mzml_children, ['cvList', 'run']))[0]
#
#         pp = {}
#         for j in seq:
#             filed = node_attr_recurse(s=root[imzml][j], d=4, c=0, ii=[])
#             dn = {}
#             for i in range(len(filed)):
#                 dn.update(dict(zip([filed[i]['path'] + '_' + re.sub('\{.*\}', '', x) for x in list(filed[i].keys())[1:]],
#                                    list(filed[i].values())[1:])))
#             pp.update({mzml_children[j]: dn})
#
#         run = root[imzml][mzml_children.index('run')]
#         # out=collect_spectra_chrom(s, ii={}, d=9, c=0, flag='1', tag='')
#         out31 = collect_spectra_chrom(s=run, ii={}, d=20, c=0, flag=flag, tag='', obos=self.obo_ids)
#
#         # summarise and collect data
#         # check how many msl1 and ms2
#
#         ms = []
#         time = []
#         cg = []
#         meta = []
#         for i in out31:
#             meta.append(out31[i]['meta'])
#             if ('s' in i) and (out31[i]['meta']['MS:1000511'] == '1'):
#                 dd = []
#                 for j in (out31[i]['data']):
#                     dd.append(out31[i]['data'][j]['d'])
#                 ms.append(dd)
#                 # time.append(np.ones_like(out31[i]['data']['m/z']['d']) * float(out31[i]['meta']['MS:1000016']))
#                 time.append(float(out31[i]['meta']['MS:1000016']))
#             if 'c' in i:
#                 dd = []
#                 for j in (out31[i]['data']):
#                     dd.append(out31[i]['data'][j]['d'])
#                 cg.append(dd)
#
#         df = pd.DataFrame(meta, index=list(out31.keys()))
#         df.columns = [self.obo_ids[x]['name'].replace(' ', '_') if x in self.obo_ids.keys() else x for x in df.columns.values]
#         df['fname'] = self.fpath
#         # summarise columns
#         summary = df.describe().T
#
#         self.dstype='Ukwn'
#         # if all chromatograms -> targeted assay
#         if all(['c' in x for x in out31.keys()]):
#             self.dstype = 'srm, nb of functions: ' + str(len(cg))
#
#             # collect chromatogram data
#             idx=np.where((df['selected_reaction_monitoring_chromatogram']==True).values)[0]
#             le = len(cg[idx[0]])
#             if le != 2:
#                 raise ValueError('Check SRM dataArray length')
#             stime = [cg[0] for i in idx]
#             Int = [cg[1] for i in idx]
#
#             return (self.dstype, df, stime, Int, summary)
#
#         else:
#             dstype = 'untargeted assay'
#             if list(out31['s1']['data'].keys())[0] == 'm/z':
#                 mz = np.concatenate([x[0] for x in ms])
#             if list(out31['s1']['data'].keys())[1] == 'Int':
#                 Int = np.concatenate([x[1] for x in ms])
#
#             size = df[df.ms_level == '1'].defaultArrayLength.astype(int).values
#             idx_time = np.argsort(time)
#             scanid = np.concatenate([np.ones(size[i]) * idx_time[i] for i in range(len(size))])
#             stime = np.concatenate([np.ones(size[i]) * time[i] for i in range(len(size))])
#
#             if len(np.unique([len(scanid), len(mz), len(Int), len(stime)])) > 1:
#                 raise ValueError('Mz, rt or scanid dimensions not matching - possibly corrupt data,')
#
#             if print_summary:
#                 print(dstype)
#                 print(summary)
#
#             return (dstype, df, scanid, mz, Int, stime, summary)
#
# class ExpSet(list_exp):
#     from ._btree import bst_search, spec_distance
#
#     def __init__(self, path,  msExp, ftype='mzml', nmax=None):
#         super().__init__(path, ftype=ftype, print_summary=True, nmax=nmax)
#
#     def read(self, pattern=None, multicore=True):
#         import os
#         import numpy as np
#         if not isinstance(pattern, type(None)):
#             import re
#             self.files = list(filter(lambda x: re.search(pattern, x), self.files))
#             print('Selected are ' + str(len(self.files)) + ' files. Start importing...')
#             self.df = self.df[self.df.exp.isin(self.files)]
#             self.df.index = ['s' + str(i) for i in range(self.df.shape[0])]
#         else: print('Start importing...')
#
#         if multicore & (len(self.files) > 1):
#             import multiprocessing
#             from joblib import Parallel, delayed
#
#             ncore = np.min([multiprocessing.cpu_count() - 2, len(self.files)])
#             print(f'Using {ncore} workers.')
#             self.exp = Parallel(n_jobs=ncore)(delayed(msExp)(os.path.join(self.path, i), '1', False) for i in self.files)
#         else:
#             # if (len(self.files) > 1):
#             #     print('Importing file')
#             # else:
#             #     print('Importing file')
#             self.exp = []
#             for f in self.files:
#                 fpath=os.path.join(self.path, f)
#                 self.exp.append(msExp(fpath, '1', False))
#
#     def get_bpc(self, plot):
#         import numpy as np
#         from scipy.interpolate import interp1d
#
#         st = []
#         st_delta = []
#         I = []
#         st_inter = []
#         for exp in self.exp:
#             df_l1 = exp.df.loc[exp.df.ms_level == '1']
#             exp_st = df_l1.scan_start_time
#             exp_I = df_l1.base_peak_intensity.astype(float)
#             I.append(exp_I)
#             st_delta.append(np.median(np.diff(exp_st)))
#             st.append(df_l1.scan_start_time)
#             st_inter.append(interp1d(exp_st, exp_I, bounds_error=False, fill_value=0))
#
#         tmin = np.concatenate(st).min()
#         tmax = np.concatenate(st).max()
#         x_new = np.arange(tmin, tmax, np.mean(st_delta))
#
#         # interpolate to common x points
#         self.bpc= np.zeros((len(self.exp), len(x_new)))
#         self.bpc_st = x_new
#         for i in range(len(st_inter)):
#             self.bpc[i] = st_inter[i](x_new)
#
#         if plot:
#             import matplotlib.pyplot as plt
#             fig = plt.figure()
#             ax = plt.subplot(111)
#             ax.plot(x_new, self.bpc.T, linewidth=0.8)
#             #ax.legend(self.df.index.values)
#
#             fig.canvas.draw()
#             offset = ax.yaxis.get_major_formatter().get_offset()
#
#             ax.set_xlabel(r"$\bfs$")
#             ax.set_ylabel(r"$\bfSum Max Count$")
#             ax2 = ax.twiny()
#
#             ax.yaxis.offsetText.set_visible(False)
#             # ax1.yaxis.set_label_text("original label" + " " + offset)
#             if offset != '': ax.yaxis.set_label_text(r"$\bfInt/count$ ($x 10^" + offset.replace('1e', '') + '$)')
#
#             def tick_conv(X):
#                 V = X / 60
#                 return ["%.2f" % z for z in V]
#
#             tick_loc = np.arange(np.round(tmin / 60), np.round(tmax / 60), 2, dtype=float) * 60
#             ax2.set_xlim(ax.get_xlim())
#             ax2.set_xticks(tick_loc)
#             ax2.set_xticklabels(tick_conv(tick_loc))
#             ax2.set_xlabel(r"$\bfmin$")
#             # ax2.yaxis.set_label_text(r"$Int/count$ " + offset)
#
#             ax.legend(self.df.index.values, loc='best', bbox_to_anchor=(0.5, 0.5, 0.5, 0.5))
#
#     def get_tic(self, plot):
#         import numpy as np
#         from scipy.interpolate import interp1d
#
#         st = []
#         st_delta = []
#         I = []
#         st_inter = []
#         for exp in self.exp:
#             df_l1 = exp.df.loc[exp.df.ms_level == '1']
#             exp_st = df_l1.scan_start_time
#             exp_I = df_l1.total_ion_current.astype(float)
#             I.append(exp_I)
#             st_delta.append(np.median(np.diff(exp_st)))
#             st.append(df_l1.scan_start_time)
#             st_inter.append(interp1d(exp_st, exp_I, bounds_error=False, fill_value=0))
#
#         tmin = np.concatenate(st).min()
#         tmax = np.concatenate(st).max()
#         x_new = np.arange(tmin, tmax, np.mean(st_delta))
#
#         # interpolate to common x points
#         self.tic= np.zeros((len(self.exp), len(x_new)))
#         self.tic_st = x_new
#         for i in range(len(st_inter)):
#             self.tic[i] = st_inter[i](x_new)
#
#         if plot:
#             import matplotlib.pyplot as plt
#             fig = plt.figure()
#             ax = plt.subplot(111)
#             ax.plot(x_new, self.tic.T, linewidth=0.8)
#             #ax.legend(self.df.index.values)
#
#             fig.canvas.draw()
#             offset = ax.yaxis.get_major_formatter().get_offset()
#
#             ax.set_xlabel(r"$\bfs$")
#             #ax.set_ylabel(r"$\bfTotal Count$")
#             ax2 = ax.twiny()
#
#             ax.yaxis.offsetText.set_visible(False)
#             # ax1.yaxis.set_label_text("original label" + " " + offset)
#             if offset != '': ax.yaxis.set_label_text(r"$\bfTotal$ $\bfInt/count$ ($x 10^" + offset.replace('1e', '') + '$)')
#
#             def tick_conv(X):
#                 V = X / 60
#                 return ["%.2f" % z for z in V]
#
#             tick_loc = np.arange(np.round(tmin / 60), np.round(tmax / 60), 2, dtype=float) * 60
#             ax2.set_xlim(ax.get_xlim())
#             ax2.set_xticks(tick_loc)
#             ax2.set_xticklabels(tick_conv(tick_loc))
#             ax2.set_xlabel(r"$\bfmin$")
#             # ax2.yaxis.set_label_text(r"$Int/count$ " + offset)
#
#             ax.legend(self.df.index.values, loc='best', bbox_to_anchor=(0.5, 0.5, 0.5, 0.5))
#
#     def chrom_btree(self, ctype='bpc', index=0, depth=4, minle=100, linkage='ward', xlab='Scantime (s)', ylab='Total Count', ax=None, colour='green'):
#         if ctype == 'bpc':
#             if not hasattr(self, ctype):
#                 self.get_bpc(plot=False)
#             mm = self.bpc[index]
#             mx = self.bpc_st
#         elif ctype == 'tic':
#             if not hasattr(self, ctype):
#                 self.get_tic(plot=False)
#             mm = self.tic[index]
#             mx = self.tic_st
#
#         self.btree = self.bst_search(mx, mm,depth, minle)
#         return self.btree.vis_graph(xlab=xlab, ylab=ylab, ax=ax, col=colour)
#
#     def chrom_dist(self, ctype='bpc', depth=4, minle=3, linkage='ward', **kwargs):
#
#
#         if ctype == 'bpc':
#             if not hasattr(self, ctype):
#                 self.get_bpc(plot=False)
#             mm = self.bpc
#             mx = self.bpc_st
#         elif ctype == 'tic':
#             if not hasattr(self, ctype):
#                 self.get_tic(plot=False)
#             mm = self.tic
#             mx = self.tic_st
#
#         self.spec_distance(mm, mx, depth, minle, linkage, **kwargs)
#
#     def pick_peaks(self, st_adj=6e2, q_noise=0.89, dbs_par={'eps': 0.02, 'min_samples': 2},
#                  selection={'mz_min': None, 'mz_max': None, 'rt_min': None, 'rt_max': None},
#                    qc_par={'st_len_min': 5, 'raggedness': 0.15, 'non_neg': 0.8, 'sId_gap': 0.15, 'sino': 2, 'ppm': 15},
#                    multicore=True):
#
#         self.qc_par = qc_par
#         import numpy as np
#         print('Start peak picking...')
#
#         def pp(ex, st_adj, q_noise, dbs_par, selection):
#             try:
#                 ex.peak_picking(st_adj, q_noise, dbs_par, selection)
#                 return ex
#             except Exception as e:
#                 return ex
#
#         self.pp_selection = selection
#
#         if multicore & (len(self.exp) > 1):
#             import multiprocessing
#             from joblib import Parallel, delayed
#             import numpy as np
#
#             ncore = np.min([multiprocessing.cpu_count() - 2, len(self.files)])
#             print(f'Using {ncore} workers.')
#             self.exp = Parallel(n_jobs=ncore)(delayed(pp)(i, st_adj, q_noise, dbs_par, selection) for i in self.exp)
#         else:
#             for i in range(len(self.exp)):
#                 ex=self.exp[i]
#                 print(ex.fpath)
#                 self.exp[i]=pp(ex, st_adj, q_noise, dbs_par, selection)
#
#     def vis_peakGroups(self):
#         import matplotlib.pyplot as plt
#
#         prop_cycle = plt.rcParams['axes.prop_cycle']
#         colors = prop_cycle.by_key()['color']
#
#         c = 0
#         d = len(colors)-1
#         for e in range(len(self.exp)):
#             ex = self.exp[e]
#             if hasattr(ex, 'feat_l3'):
#                 print(e)
#                 fl2 = ex.feat_l3
#                 rt=[]
#                 mz=[]
#                 for i in fl2:
#                     rt.append(ex.feat[i]['descr']['rt_maxI'])
#                     mz.append(ex.feat[i]['descr']['mz_maxI'])
#                 plt.scatter(rt, mz, facecolors='none', edgecolors=colors[c], label=self.df.index[c])
#
#                 if c == d:
#                     c=0
#                 else:
#                     c=c+1
#         plt.legend()
#         plt.xlabel('Scan time (s)')
#         plt.ylabel('m/z')
#
#
#
#
#
#
#     # def rt_ali(self, method='warping', parallel=True):
#     #     # import sys
#     #     # if ((sys.version_info.major > 3) | (sys.version_info.major == 3 $ sys.version_info.minor <8)):
#     #     #     raise ValueError('Update Python to 3.9 or higher.')
#     #     #
#     #
#     #     if not hasattr(self, 'bpc'):
#     #         self.get_bpc(plot=False)
#     #
#     #     from fastdtw import fastdtw
#     #     from scipy.spatial.distance import euclidean
#     #     import numpy as np
#     #     import matplotlib.pyplot as plt
#     #
#     #     self.bpc.shape
#     #     self.bpc_st.shape
#     #
#     #     x =  self.bpc[0]
#     #     y = np.mean(self.bpc, 0)
#     #     distance, path = fastdtw(x, y, dist=euclidean)
#     #     len(path)
#     #     np.arange(0, len(x))
#     #
#     #     x_new = []
#     #     y_new = []
#     #     ix_last=-1
#     #     iy_last=-1
#     #     for i in range(len(path)):
#     #         x_new.append(x[path[i][0]])
#     #         y_new.append(y[path[i][0]])
#     #
#     #     plt.plot(x_new)
#     #     plt.plot(y_new)
#     #
#     #     plt.plot(x)
#     #     plt.plot(y)
#     #
#     #
#     #         if i > 0 :
#     #             ix_last = path[i-1][0]
#     #             iy_last = path[i - 1][1]
#     #
#     #         if path[i][0] == ix_last: x_new.append(x[path[i][0]]
#     #         else: x_new.append(path[i][0])
#     #
#     #         if path[i][1] == iy_last: continue
#     #         else: y_new.append(path[i][1])
#     #
#     #
#     #     # get warping functions
#
# def rec_attrib(s, ii={}):
#     # recursively collecting cvParam
#     for at in list(s):
#         if 'cvParam' in at.tag:
#             if 'name' in at.attrib.keys():
#                 ii.update({at.attrib['name'].replace(' ', '_'): at.attrib['value']})
#         if len(list(at)) > 0:
#             rec_attrib(at, ii)
#
#     return ii
#
# def extract_mass_spectrum_lev1(i, s_level, s_dic):
#     import base64
#     import numpy as np
#     import zlib
#
#     s = s_level[i]
#     iistart = s_dic[i]
#     ii = rec_attrib(s, iistart)
#     # Extract data
#     # two binary arrays: m/z value and intensity
#     # read m/z value
#     le = len(list(s)) - 1
#     # pars = list(list(list(s)[7])[0]) # ANPC files
#     pars = list(list(list(s)[le])[0])  # NPC files
#
#     dt = arraydtype(pars)
#
#     # check if compression
#     if pars[1].attrib['name'] == 'zlib compression':
#         mz = np.frombuffer(zlib.decompress(base64.b64decode(pars[3].text)), dtype=dt)
#     else:
#         mz = np.frombuffer(base64.b64decode(pars[3].text), dtype=dt)
#
#     # read intensity values
#     pars = list(list(list(s)[le])[1])
#     dt = arraydtype(pars)
#     if pars[1].attrib['name'] == 'zlib compression':
#         I = np.frombuffer(zlib.decompress(base64.b64decode(pars[3].text)), dtype=dt)
#     else:
#         I = np.frombuffer(base64.b64decode(pars[3].text), dtype=dt)
#
#     return (ii, mz, I)
#
# def collect_spectra_chrom(s, ii={}, d=9, c=0, flag='1', tag='', obos={}):
#     import re
#     if c == d: return
#     if (('chromatogram' == tag) | ('spectrum' == tag)):
#         rtype = 'Ukwn'
#         if ('chromatogram' == tag):
#             rtype = 'c'
#         if ('spectrum' == tag):
#             rtype = 's'
#         add_meta, data = extr_spectrum_chromg(s, flag=flag, rtype=rtype, obos=obos)
#         # if rtype == 'c': print(tag); print(add_meta); print(data)
#         ii.update({rtype + add_meta['index']: {'meta': add_meta, 'data': data}})
#
#     ss = list(s)
#     if len(ss) > 0:
#         for i in range(len(ss)):
#             tag = re.sub('\{.*\}', '', ss[i].tag)
#             collect_spectra_chrom(ss[i], ii=ii, d=d, c=c + 1, flag=flag, tag=tag, obos=obos)
#
#     return ii
#
# def extr_spectrum_chromg(x, flag='1', rtype='s', obos=[]):
#     add_meta = vaPar_recurse(x, ii={}, d=6, c=0, dname='name', dval='value', ddt='all')
#
#     if rtype == 's':  # spectrum
#         if flag == '1':  # mslevel 1 input
#             if ('MS:1000511' in add_meta.keys()):  # find mslevel information
#                 if add_meta['MS:1000511'] == '1':  # read in ms1
#                     out = data_recurse1(x, ii={}, d=9, c=0, ip='', obos=obos)
#                 else:
#                     return (add_meta, [])
#             else:  # can;t find mslevel info
#                 print('MS level information not found not found - reading all experiments.')
#                 out = data_recurse1(x, ii={}, d=9, c=0, ip='', obos=obos)
#         else:
#             out = data_recurse1(x, ii={}, d=9, c=0, ip='', obos=obos)
#     else:
#         out = data_recurse1(x, ii={}, d=9, c=0, ip='', obos=obos)
#     return (add_meta, out)
#
# def data_recurse1(s, ii={}, d=9, c=0, ip='', obos=[]):
#     import re
#     # recursively extracts attributes from node s and with depth  d, add label children iterator as prefix
#     if c == d: return
#     if ('binaryDataArray' == re.sub('\{.*\}', '', s.tag)):
#         iis, ft = read_bin(s, obos)
#         ii.update({ip + ft: iis})
#     # if ('cvParam' in s.tag):
#     ss = list(s)
#     if len(ss) > 0:
#         for i in range(len(ss)):
#             tag = re.sub('\{.*\}', '', ss[i].tag)
#             if ('chromatogram' == tag):
#                 ip = 'c'
#             if ('spectrum' == tag):
#                 ip = 's'
#             data_recurse1(s=ss[i], ii=ii, d=d, c=c + 1, ip=ip, obos=obos)
#     return ii
#
# def dt_co(dvars):
#     import numpy as np
#     # dtype and compression
#     dt = None  # data type
#     co = None  # compression
#     ft = 'ukwn'  # feature type
#     if 'MS:1000523' in dvars.keys():
#         dt = np.dtype('<d')
#
#     if 'MS:1000522' in dvars.keys():
#         dt = np.dtype('<i8')
#
#     if 'MS:1000521' in dvars.keys():
#         dt = np.dtype('<f')
#
#     if 'MS:1000519' in dvars.keys():
#         dt = np.dtype('<i4')
#
#     if isinstance(dt, type(None)):
#         raise ValueError('Unknown variable type')
#
#     if 'MS:1000574' in dvars.keys():
#         co = 'zlib'
#
#     if 'MS:1000514' in dvars.keys():
#         ft = 'm/z'
#     if 'MS:1000515' in dvars.keys():
#         ft = 'Int'
#
#     if 'MS:1000595' in dvars.keys():
#         ft = 'time'
#     if 'MS:1000516' in dvars.keys():
#         ft = 'Charge'
#     if 'MS:1000517' in dvars.keys():
#         ft = 'sino'
#
#     return (dt, co, ft)
#
# def read_bin(k, obo_ids):
#     # k is binary array
#     import numpy as np
#     import zlib
#     import base64
#     # collect metadata
#     dvars = vaPar_recurse(k, ii={}, d=3, c=0, dname='accession', dval='cvRef', ddt='-')
#     dt, co, ft = dt_co(dvars)
#
#     child = children(k)
#     dbin = k[child.index('binary')]
#
#     if co == 'zlib':
#         d = np.frombuffer(zlib.decompress(base64.b64decode(dbin.text)), dtype=dt)
#     else:
#         d = np.frombuffer(base64.b64decode(dbin.text), dtype=dt)
#     # return data and meta
#     out = {'i': [{x: obo_ids[x]['name']} for x in dvars.keys() if x in obo_ids.keys()], 'd': d}
#
#     return (out, ft)
#
# def vaPar_recurse(s, ii={}, d=9, c=0, dname='accession', dval='value', ddt='all'):
#     import re
#     # recursively extracts attributes from node s and with depth  d, add label children iterator as prefix
#     if c == d: return
#     if ('cvParam' in s.tag):
#         iis = {}
#         name = s.attrib[dname]
#         value = s.attrib[dval]
#         if value == '': value = True
#         iis.update({name: value})
#         if 'unitCvRef' in s.attrib:
#             iis.update({s.attrib['unitAccession']: s.attrib['unitName']})
#         ii.update(iis)
#     else:
#         if ddt == 'all':
#             iis = s.attrib
#             ii.update(iis)
#     # if ('cvParam' in s.tag):
#     ss = list(s)
#     if len(ss) > 0:
#         # if ('spectrum' in s.tag):
#         for i in range(len(ss)):
#             vaPar_recurse(s=ss[i], ii=ii, d=d, c=c + 1, ddt=ddt)
#     return ii
#
# def get_obo(obos, obo_ids={}):
#     # download obo annotation data
#     # create single dict with keys being of obo ids, eg, MS:1000500
#     # obos is first cv element in mzml node
#     import obonet
#     for o in obos:
#         id=o.attrib['id']
#         if id not in obo_ids.keys():
#             gr = obonet.read_obo(o.attrib['URI'])
#             gv = gr.nodes(data=True)
#             map = {id_: {'name': data.get('name'), 'def': data.get('def'), 'is_a': data.get('is_A')} for
#                           id_, data in gv}
#             #obo_ids.update({id: {'inf': o.attrib, 'data': map}})
#             obo_ids.update(map)
#
#     return obo_ids
#
# def children(xr):
#     import re
#     return [re.sub('\{.*\}', '', x.tag) for x in list(xr)]
#
# def exp_meta(root, ind, d):
#     # combine attributes to dataframe with column names prefixed acc to child/parent relationship
#     # remove web links in tags
#     import pandas as pd
#     import re
#     filed = node_attr_recurse(s=root, d=d, c=0, ii=[])
#
#     dn = {}
#     for i in range(len(filed)):
#         dn.update(dict(zip([filed[i]['path'] + '_' + re.sub('\{.*\}', '', x) for x in list(filed[i].keys())[1:]],
#                            list(filed[i].values())[1:])))
#     return pd.DataFrame(dn, index=[ind])
#
# def node_attr_recurse(s, ii=[], d=3, c=0,  pre=0):
#     import re
#     # recursively extracts attributes from node s and with depth  d, add label children iterator as prefix
#     if c == d: return
#     # define ms level
#     if ('list' not in s.tag) | ('spectrum' not in s.tag):
#         iis = {}
#         iis['path'] = re.sub('\{.*\}', '', s.tag) + '_' + str(pre)
#         iis.update(s.attrib)
#         ii.append(iis)
#     if len(list(s)) > 0:
#         if ('spectrum' not in s.tag.lower()):
#             ss = list(s)
#             for i in range(len(list(ss))):
#                 at = ss[i]
#                 node_attr_recurse(s=at, ii=ii,d=d, c=c + 1, pre=i)
#     return ii
#
# def read_exp(path, ftype='mzML', plot_chrom=False, n_max=1000, only_summary=False, d=4):
#     import subprocess
#     import xml.etree.ElementTree as ET
#
#     cmd = 'find ' + path + ' -type f -iname *'+ ftype
#     sp = subprocess.getoutput(cmd)
#     files = sp.split('\n')
#
#     if len(files) == 0:
#         raise ValueError(r'No .' + ftype + ' files found in path: '+path)
#
#     if len(files) > n_max:
#         files=files[0:n_max]
#
#     if only_summary:
#         print('Creating experiment summary.')
#         #import hashlib
#         #hashlib.md5(open(fid,'rb').read()).hexdigest()
#
#         df_summary = []
#         for i in range(len(files)):
#             print(files[i])
#             tree = ET.parse(files[i])
#             root = tree.getroot()
#             #tt = exp_meta(root=root, ind=i, d=d).T
#             df_summary.append(exp_meta(root=root, ind=i, d=d))
#
#         df = pd.concat(df_summary).T
#         return df
#     else:
#
#         data=[]
#         df_summary = []
#         print('Importing '+ str(len(files)) + ' files.')
#         for i in range(len(files)):
#             tree = ET.parse(files[i])
#             root = tree.getroot()
#             df_summary.append(exp_meta(root=root, ind=i, d=d))
#             out = read_mzml_file(root, ms_lev=1, plot=plot_chrom)
#             data.append(out[0:3])
#
#         df = pd.concat(df_summary).T
#         return (df, data)
#
# def read_mzml_file(root, ms_lev=1, plot=True):
#     from tqdm import tqdm
#     import time
#     import numpy as np
#     import pandas as pd
#     # file specific variables
#     le = len(list(root[0]))-1
#     id = root[0][le].attrib['id']
#
#
#     StartTime=root[0][le].attrib['startTimeStamp'] # file specific
#     #ii=root[0][6][0].attrib
#
#     # filter ms level spectra
#     s_all = list(root[0][le][0])
#
#     s_level=[]
#     s_dic=[]
#     for s in s_all:
#         scan = s.attrib['id'].replace('scan=', '')
#         arrlen = s.attrib['defaultArrayLength']
#
#         ii = {'fname': id, 'acquTime': StartTime, 'scan_id': scan, 'defaultArrayLength': arrlen}
#         if 'name' in s[0].attrib.keys():
#             if ((s[0].attrib['name'] == 'ms level') & (s[0].attrib['value'] == str(ms_lev))) | (s[0].attrib['name'] == 'MS1 spectrum'):
#                 # desired ms level
#                 s_level.append(s)
#                 s_dic.append(ii)
#     print('Number of MS level '+ str(ms_lev)+ ' mass spectra : ' + str(len(s_level)))
#     # loop over spectra
#     # # check multicore
#     # import multiprocessing
#     # from joblib import Parallel, delayed
#     # from tqdm import tqdm
#     # import time
#     # ncore = multiprocessing.cpu_count() - 2
#     # #spec_seq = tqdm(np.arange(len(s_level)))
#     # spec_seq = tqdm(np.arange(len(s_level)))
#     # a= time.time()
#     # bls = Parallel(n_jobs=ncore, require='sharedmem')(delayed(extract_mass_spectrum_lev1)(i, s_level, s_dic) for i in spec_seq)
#     # b=time.time()
#     # extract mz, I ,df
#     # s1=(b-a)/60
#
#     spec_seq = np.arange(len(s_level))
#     #spec_seq = np.arange(10)
#     ss = []
#     I = []
#     mz = []
#     a = time.time()
#     for i in tqdm(spec_seq):
#         ds, ms, Is = extract_mass_spectrum_lev1(i, s_level, s_dic)
#         ss.append(ds)
#         mz.append(ms)
#         I.append(Is)
#
#     b = time.time()
#     s2 = (b - a) / 60
#
#     df = pd.DataFrame(ss)
#
#     swll = df['scan_window_lower_limit'].unique()
#     swul = df['scan_window_upper_limit'].unique()
#
#     if len(swll) ==1 & len(swul) == 1:
#         print('M/z scan window: ', swll[0] + "-" + swul[0] + ' m/z')
#     else:
#         print('Incosistent m/z scan window!')
#         print('Lower bound: '+str(swll)+' m/z')
#         print('Upper bound: '+str(swul)+' m/z')
#
#
#     #df.groupby('scan_id')['scan_start_time'].min()
#
#     df.scan_start_time=df.scan_start_time.astype(float)
#
#     tmin=df.scan_start_time.min()
#     tmax=df.scan_start_time.max()
#     print('Recording time: ' + str(np.round(tmin,1)) + '-' + str(np.round(tmax,1)), 's (' + str(np.round((tmax-tmin)/60, 1)), 'm)')
#     print('Read-in time: ' + str(np.round(s2, 1)) + ' min')
#     # df.scan_id = df.scan_id.astype(int)
#     df.scan_id = np.arange(1, df.shape[0]+1)
#     df.total_ion_current=df.total_ion_current.astype(float)
#     df.base_peak_intensity=df.base_peak_intensity.astype(float)
#
#     # if plot:
#     #     pl = plt_chromatograms(df, tmin, tmax)
#     #     return (mz, I, df, pl)
#     # else:
#     return (mz, I, df)
#
# def arraydtype(pars):
#     import numpy as np
#     # determines base64 single or double precision, all in little endian
#     pdic = {}
#     for pp in pars:
#         if 'name' in pp.attrib.keys():
#             pdic.update({pp.attrib['accession'].replace(' ', '_'): pp.attrib['name']})
#         else:
#             pdic.update(pp.attrib)
#
#     dt = None
#     if 'MS:1000523' in pdic.keys():
#         dt = np.dtype('<d')
#
#     if 'MS:1000522' in pdic.keys():
#         dt = np.dtype('<i8')
#
#     if 'MS:1000521' in pdic.keys():
#         dt = np.dtype('<f')
#
#     if 'MS:1000519' in pdic.keys():
#         dt = np.dtype('<i4')
#
#     if isinstance(dt, type(None)):
#         raise ValueError('Unknown variable type')
#
#     return dt
