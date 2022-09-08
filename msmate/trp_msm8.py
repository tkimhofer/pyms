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
import collections
# import itertools
sys.path.append('/Users/tkimhofer/pyt/pyms')
# import msmate as ms
from msmate.helpers import _collect_spectra_chrom, _get_obo, _children, _node_attr_recurse
import re, os

# reading experiment annotation data that defines MRM's
class SIR:
    # reads single ion reaction from targeted assay
    def __init__(self, d):
        self.massPrecursor = d['SIRMass']
        self.massProduct = d['SIRMass_2_']

        if (self.massPrecursor != d["Mass(amu)_"]) | (self.massProduct != d["Mass2(amu)_"]):
            raise Warning('diverging SIR and amu masses')
        self.autoDwell = d['SIRAutoDwell']
        self.autoDwellTime = d['SIRAutoDwell']
        self.delay = d['SIRDelay']
        self.useAsLockMass = d['UseAsLockMass']
        self.dwell_s = d['Dwell(s)_']
        self.conevoltageV = d['ConeVoltage(V)_']
        self.collisionEnergy = d['CollisionEnergy(V)_']
        self.interchanneldelay = d['InterChannelDelay(s)_']
        self.compoundName = d['CompoundName_']
        self.compoundFormula = d['CompoundFormula_']

    def __repr__(self):
        stf = f"{self.compoundName}: {self.massPrecursor} m/z ---> {self.massProduct} m/z\n"
        return stf


class MRMFunction:
    # defines SIRs and parameters of multiple reaction monitoring for targeted assay
    def __init__(self, d):
        self.ftype = d['FunctionType']
        self.polarity = d['FunctionPolarity']
        self.stStart_min = d['FunctionStartTime(min)']
        self.stEnd_min = d['FunctionEndTime(min)']

        self.reactions = {k: SIR(s) for k, s in d['sim'].items()}
        self.name = '/'.join(list(set([dd.compoundName for k, dd in self.reactions.items()])))

        self.other = d

    def __repr__(self):
        ssr = "\n\t".join([f"{r.massPrecursor} m/z ---> {r.massProduct} m/z" for k, r in self.reactions.items()])
        return f"{self.name}\n\t" + ssr + '\n'


class ReadFun:
    # reads function data for targeted assay, comprising of individuals MRM functions
    def __init__(self, epath):
        import re
        pars = {}
        with open(epath, 'rb') as f:
            for t in f.readlines():
                td = t.strip().decode('latin1')
                if bool(td.isupper()):
                    feat = td
                    pars[feat] = {}
                    pars[feat]['sim'] = {}
                    continue
                if ',' in td:
                    add = td.split(',')
                    if bool(re.findall('[0-9]$', add[0])) and 'FUN' in feat:
                        key, featid, _ = re.split('([0-9])$', add[0])
                        if featid not in pars[feat]['sim']:
                            pars[feat]['sim'][featid] = {}
                        pars[feat]['sim'][featid][key] = add[1]
                    else:
                        pars[feat][add[0]] = add[1]

        self.generalInfo = pars['GENERAL INFORMATION']
        self.funcs = {k: MRMFunction(x) for k, x in pars.items() if 'FUN' in k}

    def __repr__(self):
        return f"{len(self.funcs)} functions"


epath = '/Users/TKimhofer/Downloads/Torben19Aug.exp'
efun = ReadFun(epath)


class ReadW():
    # convet to mzml, then import
    def __init__(self, dpath: str, docker: dict = {'repo': 'chambm/pwiz-skyline-i-agree-to-the-vendor-licenses:latest'}, convert:bool = False):

        """Import of targeted data from Waters experiments

           Conversion from .raw to .mzml will be performed with the official msconvert docker image (proteowiz). Note that this class is designed for use with `MsExp`.

           Args:
               dpath: Path `string` variable pointing to Bruker experiment folder (.d)
               convert: `Bool` indicating if conversion should take place using docker container inf `.p` is not present
               docker: `dict` with docker information: `{'repo': 'convo:v1', 'mslevel': 0, 'smode': [0, 1, 2, 3, 4, 5], 'amode': 2, 'seg': [0, 1, 2, 3], }`
        """
        import re
        import datetime as dt
        self.mslevel = 'targeted'
        self.convert = convert
        self.docker = {'repo': docker['repo']}

        # dpath can be raw or mzml
        if bool(re.findall("mzML$", dpath)):
            self.mzmlfn = os.path.abspath(dpath)
            self.dpath = os.path.abspath(re.sub('mzML$', 'raw', dpath))
        elif bool(re.findall("raw$", dpath)):
            self.dpath = os.path.abspath(dpath)
            self.mzmlfn = os.path.abspath(re.sub('raw$', 'mzML', dpath))
        self.msmfile = os.path.join(re.sub('mzML$', 'raw', self.dpath), f'mm8v3_edata.p')
        self.fname = os.path.basename(self.dpath)

        self.edf = {}
        try:
            f1 = os.path.join(self.dpath, '_INLET.INF')
            iadd = self._pInlet(f1)
            self.edf.update(iadd)
        except:
            pass
        try:
            f1 = os.path.join(self.dpath, '_HEADER.TXT')
            hadd = self._pInlet(f1)
            self.edf.update(hadd)
        except:
            pass

        self.aDt = dt.datetime.strptime(self.edf['Acquired Date']+self.edf['Acquired Time'], ' %d-%b-%Y %H:%M:%S')

        self._importData()


        # self.edf = pd.DataFrame(ddf, index=[0]).T


    def _importData(self):

        if (not self.convert):

            if os.path.exists(self.msmfile):
                # print('read bin')
                self. _readmsm8()
            elif os.path.exists(self.mzmlfn):
                # print('read mzml')
                self._read_mzml()
                self._createSpectMat()
                self._savemsm8()
            else:
                # print('not converted before')
                self._convoDocker()
                self._read_mzml()
                self._createSpectMat()
                self._savemsm8()
        else:
            self._convoDocker()
            self._read_mzml()
            self._createSpectMat()
            self._savemsm8()


    def _readmsm8(self):
        try:
            ss = pickle.load(open(self.msmfile, 'rb'))
            # print(ss.keys())
            self.edf = ss['edf']
            self.dfd = ss['dfd']
            self.xrawd = ss['xrawd']
        except:
            raise ValueError('Can not open binary')

    def _savemsm8(self):
        try:
            with open(self.msmfile, 'wb') as fh:
                pickle.dump({'dfd': self.dfd, 'xrawd': self.xrawd, 'edf': self.edf}, fh)
        except:
            print('Cant save msm8 binary - check disk permissions')

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
        ec = f'docker run -i --rm -e WINEDEBUG=-all -v "{os.path.dirname(self.dpath)}":/data {self.docker["repo"]} wine msconvert "/data/{self.fname}"'
        self.ex = ec
        self.rrr = os.system(ec)
        t1 = time.time()
        print(f'Conversion time: {np.round(t1 - t0)} sec')
        self.mzmlfn = re.sub('raw$', 'mzML', self.dpath)
        self.fpath = re.sub('raw$', 'mzML', self.dpath)

    def _read_mzml(self):
        """Extracts MS data from mzml file"""
        # this is for mzml version 1.1.0
        # schema specification: https://raw.githubusercontent.com/HUPO-PSI/mzML/master/schema/schema_1.1/mzML1.1.0.xsd
        # read in data, files index 0 (see below)
        # flag is 1/2 for msLevel 1 or 2
        tree = ET.parse(self.mzmlfn )
        root = tree.getroot()
        child = _children(root)
        imzml = child.index('mzML')
        mzml_children = _children(root[imzml])
        obos = root[imzml][mzml_children.index('cvList')]  # controlled vocab (CV
        self.obo_ids = _get_obo(obos, obo_ids={})
        seq = np.where(~np.isin(mzml_children, ['cvList', 'run']))[0]
        pp = {}
        for j in seq:
            filed = _node_attr_recurse(s=root[imzml][j], d=4, c=0, ii=[])
            dn = {}
            for i in range(len(filed)):
                dn.update(
                    dict(zip([filed[i]['path'] + '_' + re.sub('\{.*\}', '', x) for x in list(filed[i].keys())[1:]],
                             list(filed[i].values())[1:])))
            pp.update({mzml_children[j]: dn})
        run = root[imzml][mzml_children.index('run')]
        self.out31 = _collect_spectra_chrom(s=run, ii={}, d=20, c=0, flag=self.mslevel, tag='', obos=self.obo_ids)

    def prep_df(self, df):
        if 'MS:1000127' in df.columns:
            add = np.repeat('centroided', df.shape[0])
            add[~(df['MS:1000127'] == True)] = 'profile'
            df['MS:1000127'] = add
        if 'MS:1000128' in df.columns:
            add = np.repeat('profile', df.shape[0])
            add[~(df['MS:1000128'] == True)] = 'centroided'
            df['MS:1000128'] = add

        df = df.rename(
            columns={'UO:0000010': 'time_unit', 'UO:0000031': 'time_unit', 'defaultArrayLength': 'n',
                     'MS:1000129': 'polNeg', 'MS:1000130': 'polPos',
                     'MS:1000127': 'specRepresentation', 'MS:1000128': 'specRepresentation',
                     'MS:1000505': 'MaxIntensity', 'MS:1000285': 'SumIntensity',
                     'MS:1000016': 'Rt'
                     })
        df.columns = [self.obo_ids[x]['name'].replace(' ', '_') if x in self.obo_ids.keys() else x for x in
                      df.columns.values]
        df['fname'] = self.dpath
        return df

    @staticmethod
    def _pInlet(ipath):
        # extract inlet and header information
        pars = {}
        prefix = ''
        with open(ipath, 'rb') as f:
            for t in f.readlines():
                td = t.strip().decode('latin1')
                if bool(re.findall('^-- ', td)):
                    if not 'END' in td:
                        prefix = td.replace('-', '').lstrip().rstrip() + '_'
                    else:
                        prefix = ''

                if ':' in td:
                    add = td.replace('$$ ', '').split(':', 1)
                    pars[prefix + add[0]] = add[1]

        return pars

    def _createSpectMat(self):
        # tyr targeted data has two dimensions: 1. rt, 2. intensity
        """Organise raw MS data and scan metadata"""

        def srmSic(id):
            ss = re.split('=| ', id)
            return {'q1': ss[-7], 'q3': ss[-5], 'fid': ss[-3], 'offset': ss[-1]}


        sc_msl1 = [(i, len(x['data']['Int']['d'])) for i, x in self.out31.items() if
                   len(x['data']) == 2]  # sid of ms level 1 scans

        self.df = pd.DataFrame([x['meta'] for i, x in self.out31.items() if len(x['data']) == 2])
        self.df['defaultArrayLength'] = self.df['defaultArrayLength'].astype(int)

        from collections import defaultdict, Counter
        xrawd = {}
        if 'MS:1000129' in self.df.columns:
            nd = self.df['defaultArrayLength'][self.df['MS:1000129'] == True].sum()
            xrawd['1N'] = np.zeros((4, nd))

        if 'MS:1000130' in self.df.columns:
            nd = self.df['defaultArrayLength'][self.df['MS:1000130'] == True].sum()
            xrawd['1P'] = np.zeros((4, nd))

        row_counter = Counter({'1P': 0, '1N': 0})
        sid_counter = Counter({'1P': 0, '1N': 0})
        dfd = defaultdict(list)
        for i, s in enumerate(sc_msl1):
            d = self.out31[s[0]]
            if s[1] != self.df['defaultArrayLength'].iloc[i]:
                raise ValueError('Check data extraction')

            cbn = [[d['meta']['index']] * s[1]]
            for k in d['data']:
                cbn.append(d['data'][k]['d'])

            if 'MS:1000129' in d['meta']:
                fstr = '1N'
            elif 'MS:1000130' in d['meta']:
                fstr = '1P'

            cbn.append([sid_counter[fstr]] * s[1])
            add = np.array(cbn)
            xrawd[fstr][:, row_counter[fstr]:(row_counter[fstr] + s[1])] = add

            dfd[fstr].append(d['meta'])

            sid_counter[fstr] += 1
            row_counter[fstr] += (s[1])

        for k, d, in dfd.items(): # for each polarity
            for j in range(len(d)): # for each sample

                out = srmSic(d[j]['id'])
                dfd[k][j].update(out)

            dfd[k] = self.prep_df(pd.DataFrame(dfd[k]))

        self.dfd = dfd
        self.xrawd = xrawd
        # self.ms0string = row_counter.most_common(1)[0][0]
        # self.ms1string = None

class Trp:
    @classmethod
    def waters(cls,  dpath: str, docker: dict = {'repo': 'chambm/pwiz-skyline-i-agree-to-the-vendor-licenses:latest'}, convert:bool = False, efun=None):
        da = ReadW(dpath, docker, convert)
        return cls(dpath=da.dpath, xrawd=da.xrawd, dfd=da.dfd, edf =da.edf, efun=efun, dt=da.aDt)

    def __init__(self, dpath, xrawd, dfd, edf, efun, dt):
        self.mslevel = 'targeted'
        self.dpath = os.path.abspath(dpath)
        self.fname = os.path.basename(dpath)
        self.xrawd = xrawd
        self.dfd = dfd
        self.edf =edf
        self.efun = efun
        self.aDt = dt

    def quant(self, xd, **xargs):
        # peak fitting
        from scipy.signal import find_peaks
        peaks, heights = find_peaks(xd[2], **xargs)
        return peaks, heights
        # print(len(peaks))

    def extractFunData(self, id):
        ff = self.efun.funcs['FUNCTION ' + str(id)]
        p = '1P' if ff.polarity == 'Positive' else '1N'
        sub = self.dfd[p][self.dfd[p]['fid'] == str(id)]
        React = list(self.efun.funcs['FUNCTION ' + str(id)].reactions)
        nReact = len(React)

        # xd = self.xrawd[p][:, self.xrawd[p][0] == float(sub.fid.values[0])]
        ret = {'d': [], 'm': []}
        for i in range(nReact):
            sidx = sub['index'].iloc[i]
            ret['d'].append(self.xrawd[p][:, self.xrawd[p][0] == float(sidx)])
            ret['m'].append(sub.iloc[i])
        return ret

    def plotEfun(self, id, **xargs):
        import matplotlib.pyplot as plt

        ff = self.efun.funcs['FUNCTION ' + str(id)]
        p='1P' if ff.polarity == 'Positive' else '1N'
        sub = self.dfd[p][ self.dfd[p]['fid'] == str(id) ]
        React = list(self.efun.funcs['FUNCTION ' + str(id)].reactions)
        nReact = len(React)
        if nReact != len(sub.index.values):
            print('Product ion missing')
        if nReact == 1:
            i=0
            xd = self.xrawd[p][:,  self.xrawd[p][0] == float(sub.fid.values[0]) ]
            fig, axs = plt.subplots(nReact, sharex=True, sharey=True)
            axs.plot(xd[1], xd[2])
            pidx, pint = self.quant(xd, **xargs)
            axs.scatter(xd[1][pidx], pint['peak_heights'], c='red')
            axs.set_ylabel(r"$\bfCount$")
            axs.text(0.77, 0.9, f'Pr@ {ff.reactions[str(i + 1)].massProduct} m/z', rotation=0, fontsize=8,
                        transform=axs.transAxes,
                        ha='left')
            axs.text(0.77, 0.8, f'st:  {ff.stEnd_min}-{ff.stStart_min} min', rotation=0, fontsize=8,
                        transform=axs.transAxes,
                        ha='left')
            axs.text(0.77, 0.7, f'{ff.reactions[str(i + 1)].compoundName}', rotation=0, fontsize=8,
                        transform=axs.transAxes,
                        ha='left')
            axs.text(0.77, 0.5, f'Co-E: {ff.reactions[str(i + 1)].collisionEnergy} eV', rotation=0, fontsize=8,
                        transform=axs.transAxes,
                        ha='left')
            axs.text(0.77, 0.4, f'Conevolt: {ff.reactions[str(i + 1)].conevoltageV}', rotation=0, fontsize=8,
                        transform=axs.transAxes,
                        ha='left')
            axs.text(0.77, 0.3, f'Polarity: {ff.polarity}', rotation=0, fontsize=8,
                        transform=axs.transAxes,
                        ha='left')
            axs.text(0.77, 0.2, f'P@ {ff.reactions[str(i + 1)].massPrecursor} m/z', rotation=0, fontsize=8,
                        transform=axs.transAxes,
                        ha='left')
            fig.suptitle(f"F{str(id)}:  {ff.name} @ {ff.reactions[str(i + 1)].massPrecursor} m/z")
            axs.text(1.04, 0, self.fname, rotation=90, fontsize=6, transform=axs.transAxes)
            axs.set_xlabel(r"$\bfScantime (min)$")
        else:
            fig, axs = plt.subplots(nReact, sharex=True, sharey=True)
            for i in range(nReact):
                sidx = sub['index'].iloc[i]
                xd =  self.xrawd[p][:, self.xrawd[p][0] == float(sidx)]
                axs[i].plot(xd[1], xd[2])
                pidx, pint = self.quant(xd, **xargs)
                axs[i].scatter(xd[1][pidx], pint['peak_heights'], c='red')
                axs[i].set_ylabel(r"$\bfCount$")
                print(self.efun.funcs['FUNCTION ' + str(id)].reactions[str(i + 1)])
                axs[i].text(0.77, 0.9, f'Pr@ {ff.reactions[str(i + 1)].massProduct} m/z', rotation=0, fontsize=8, transform=axs[i].transAxes,
                            ha='left')
                axs[i].text(0.77, 0.8, f'st:  {ff.stEnd_min}-{ff.stStart_min} min', rotation=0, fontsize=8,
                            transform=axs[i].transAxes,
                            ha='left')
                axs[i].text(0.77, 0.7, f'{ff.reactions[str(i + 1)].compoundName}', rotation=0, fontsize=8,
                            transform=axs[i].transAxes,
                            ha='left')
                axs[i].text(0.77, 0.5, f'Co-E: {ff.reactions[str(i + 1)].collisionEnergy} eV', rotation=0, fontsize=8,
                            transform=axs[i].transAxes,
                            ha='left')
                axs[i].text(0.77, 0.4, f'Conevolt: {ff.reactions[str(i + 1)].conevoltageV}', rotation=0, fontsize=8,
                            transform=axs[i].transAxes,
                            ha='left')
                axs[i].text(0.77, 0.3, f'Polarity: {ff.polarity}', rotation=0, fontsize=8,
                            transform=axs[i].transAxes,
                            ha='left')
                axs[i].text(0.77, 0.2, f'P@ {ff.reactions[str(i + 1)].massPrecursor} m/z', rotation=0, fontsize=8,
                            transform=axs[i].transAxes,
                            ha='left')
                fig.suptitle(f"F{str(id)}:  {ff.name} @ {ff.reactions[str(i + 1)].massPrecursor} m/z")

            axs[i].text(1.04, 0, self.fname, rotation=90, fontsize=6, transform=axs[i].transAxes)
            axs[i].set_xlabel(r"$\bfScantime (min)$")

ss=Trp.waters('/Volumes/ANPC_ext1/trp_qc/RCY_TRP_023_Robtot_Spiking_04Aug2022_DB_66.raw', convert=False, efun=efun)
ss.plotEfun(id=6, prominence=10000, height=800, width=5)
out=ss.extractFunData(id='30')


# define experiment set for Trp data (import series of trp experiments)
# plot functions, QC and quantify signals
class ExpSet:
    def __init__(self, path='/Volumes/ANPC_ext1/trp_qc/', convert=False):
        import re, os
        import numpy as np
        # find all waters experiment files
        # try to annotate sample type (LTR vs Cal vs Sample)
        self.exp= []
        # self.calib = []
        self.df = {}
        self.areas = {}
        c = 0
        for d in os.listdir(path):
            if bool(re.findall(".*raw$", d)):
            #     print(d)
                try:
                    c += 1
                    efile = Trp.waters(os.path.join(path, d), efun=efun, convert=False)
                    if '2022_Cal' in efile.fname:
                        efile.stype = 'Calibration'
                        efile.cpoint = efile.fname.split('_')[-2]
                        self.exp.append(efile)
                        self.df.update({efile.fname: {'stype': efile.stype , 'cpoint': efile.cpoint, 'dt': efile.aDt}})
                    else:
                        self.exp.append(efile)
                        efile.stype = 'Sample'
                        self.df.update({efile.fname: {'stype': efile.stype,  'dt': efile.aDt}})
                    self.areas[efile.fname] = {}
                    for f in efile.efun.funcs:  # for every function
                        id = f.replace('FUNCTION ', '')
                        try:
                            efDat = efile.extractFunData(id)
                            for i, xd in enumerate(efDat['d']):
                                # xd = fd
                                try:
                                    self.areas[efile.fname].update({f + '_' + str(i): pbound(xd, plot=False)})
                                except:
                                    pass
                        except:
                            pass

                except:
                    print(f'failed for file {d}')

        print(f'Imported a total of {c} files')

        dt = [x['dt'] for k, x in self.df.items()]
        ro = np.argsort(dt)

        dd = pd.DataFrame(self.df).T
        dd['RunOrder'] = ro
        dd['Sample'] = dd.index
        dd = dd[['RunOrder', 'Sample', 'stype', 'cpoint', 'dt']]
        self.exp = [self.exp[i] for i in ro]
        dd = dd.sort_values('RunOrder')
        self.df = dd

    def vis(self):
        import plotly.graph_objects as go
        fig = go.Figure()
        areas = {}
        for rec in self.exp:
            areas[rec.fname] = {}
            for f in rec.efun.funcs:  # for every function
                id = f.replace('FUNCTION ', '')
                try:
                    efDat = rec.extractFunData(id)
                    for i, fd in enumerate(efDat['d']):
                        print(i)
                        xd = fd
                        try:
                            areas[rec.fname].update({f + '_' + str(i): pbound(xd, plot=False)})
                        except:
                            pass
                except:
                    pass


es = ExpSet()
f1=es.exp[0].efun.funcs['FUNCTION 1']
f1.




        # establish run order from acquisition data (in _INIT/_HEADER file)





        t0 = pd.DataFrame([(x.aDt, x.fname) for x in ee])
        t1 = t0[1].str.split('_', expand=True)
        t2 = pd.concat([t0, t1], axis=1)

        plt.scatter([x.aDt for x in ee], np.arange(len(ee)))
    def plotFun(self, id='4', iid=0, **xargs):
        {k: x.name for k, x in self.eres[0].efun.funcs.items()}
        self.eres[0].plotEfun('26', prominence=10000, height=800, width=5)
        # , prominence=10000, height=800, width=5
        import plotly.graph_objects as go

        fig = go.Figure()

        areas = {}
        for rec in self.eres:
            areas[rec.fname]={}
            for f in rec.efun.funcs: # for every function
                print(f)
                id = f.replace('FUNCTION ', '')
                print(id)
                try:
                    efDat = rec.extractFunData(id)
                    for i, fd in enumerate(efDat['d']):
                        print(i)
                        xd = fd
                        try:
                            areas[rec.fname].update({f+'_'+str(i):  pbound(xd, plot=False)})
                        except:
                            pass
                except:
                    pass


        # mapping of functions
            map = {'a1': {'std': {'FUNCTION 5': 'Picolinic acid-D3'}, 'analyte': {'FUNCTION 4': 'Picolinic/nicotinic acid/Nicolinic acid'}},
                   'a2': {'std': {'FUNCTION 6': 'Nicotinic acid-D4'}, 'analyte': {'FUNCTION 4': 'Picolinic/nicotinic acid/Nicolinic acid'}},
                   'a3': {'std': {'FUNCTION 10': '3-HAA-D3'}, 'analyte': {'FUNCTION 8': '3-HAA'}},
                   'a4': {'std': {'FUNCTION 11': 'Dopamine-D4'}, 'analyte': {'FUNCTION 9': 'Dopamine'}},
                   'a5': {'std': {'FUNCTION 20': 'Serotonin-d4'}, 'analyte': {'FUNCTION 12': 'Serotonin'}},
                   'a6': {'std': {'FUNCTION 14': 'Tryptamine-d4'}, 'analyte': {'FUNCTION 13': 'Tryptamine'}},
                   'a7': {'std': {'FUNCTION 17': 'Quinolinic acid-D3'}, 'analyte': {'FUNCTION 15': 'Quinolinic acid'}},
                   'a8': {'std': {'FUNCTION 19': 'I-3-AA-D4'}, 'analyte': {'FUNCTION 18': 'I-3-AA'}},
                   'a9': {'std': {'FUNCTION 27': 'Kynurenic acid-D5'}, 'analyte': {'FUNCTION 22': 'Kynurenic acid'}},
                   'a10': {'std': {'FUNCTION 28': '5-HIAA-D5'}, 'analyte': {'FUNCTION 26': '5-HIAA'}},
                   'a11': {'std': {'FUNCTION 35': 'Tryptophan-d5'}, 'analyte': {'FUNCTION 30': 'Tryptophan'}},
                   'a12': {'std': {'FUNCTION 33': 'Xanthurenic acid-D4'}, 'analyte': {'FUNCTION 31': 'Xanthurenic acid'}},
                   'a13': {'std': {'FUNCTION 36': 'Kynurenine-D4'}, 'analyte': {'FUNCTION 32': 'Kynurenine'}},
                   'a14': {'std': {'FUNCTION 39': '3-HK 13C15N'}, 'analyte': {'FUNCTION 38': '3-HK'}},
                   'a15': {'std': {'FUNCTION 41': 'Neopterin-13C5'}, 'analyte': {'FUNCTION 40': 'Neopterin'}},
                   'a16': {'analyte': {'FUNCTION 3': '2-aminophenol'}},
                   'a17': {'analyte': {'FUNCTION 42': '3'}},
                   'a18': {'analyte': {'FUNCTION 16': '3-methoxy-p-tyramine'}},
                   'a19': {'analyte': {'FUNCTION 34': '4-hydroxyphenylacetylglycine'}},
                   'a20': {'analyte': {'FUNCTION 23': '5-ME_TRYT'}},
                   'a21': {'analyte': {'FUNCTION 37': '5-OH-tryptophan'}},
                   'a22': {'analyte': {'FUNCTION 21': '(-)-Epinephrine'}},
                   'a23': {'analyte': {'FUNCTION 29': 'L-Dopa'}},
                   'a24': {'analyte': {'FUNCTION 25': 'L-tryptophanol'}},
                   'a25': {'analyte': {'FUNCTION 43': 'N-acetyl-L-tyrosine'}},
                   'a26': {'analyte': {'FUNCTION 24': 'N-methylseotonin'}},
                   'a27': {'analyte': {'FUNCTION 1': 'Trimethylamine'}},
                   'a28': {'analyte': {'FUNCTION 2': 'Trimethylamine-N-oxide'}},
                   'a29': {'analyte': {'FUNCTION 7': 'Tyramine'}},
                   }

        sr = Trp.waters(rawf, efun=efun)
        sr.plotEfun(id=4, prominence=[None, 0.2], height=0.1, width=1)

        for i in range(43):
            print(i + 1)
            sr.plotEfun(id=6, prominence=10000, height=800, width=5)

        # generate calibration curves for 15 compounds where possible
        for k, d in map.items():
            # get number of SIR
            fid = list(d['analyte'].keys())[0]
            dfsub=df.iloc[:,df.columns.str.contains(fid+'_')]



eset = ExpSet()













# plot function
sys.path.append('/Users/tkimhofer/pyt/pyms')
dpath='/Volumes/ANPC_ext1/trp_qc/NW_TRP_023_manual_Spiking_05Aug2022_Cal2_10.raw'
rawf='/Volumes/ANPC_ext1/trp_qc/NW_TRP_023_manual_Spiking_05Aug2022_Cal2_10.raw'
rawf=dpath='/Volumes/ANPC_ext1/trp_qc/RCY_TRP_023_Robtot_Spiking_04Aug2022_DB_66.raw'
rawf = '/Volumes/ANPC_ext1/trp_qc/RCY_TRP_023_Robtot_Spiking_04Aug2022_SER_unhealthy_Cal2_82.raw'
rawf = '/Volumes/ANPC_ext1/trp_qc/RCY_TRP_023_Robtot_Spiking_04Aug2022_Cal8_133.raw'
sr = Trp.waters(rawf, efun=efun)
self=sr
self.plotEfun(id=4, prominence=[None, 0.2], height=0.1, width=1)

for i in range(43):
    print(i+1)
    sr.plotEfun(id=6, prominence=10000, height=800, width=5)

# read in series of trp experiments and combine lineshapes
class ExpSet:
    def __init__(self, path='/Volumes/ANPC_ext1/trp_qc/', convert=False):
        import re, os
        # find all waters experiment files
        self.eres=[]
        for d in os.listdir(path):
            if bool(re.findall(".*2_Cal.*raw$", d)):
                print(d)
                try:
                    self.eres.append(Trp.waters(os.path.join(path, d), efun=efun, convert=False))
                except:
                    print(f'failed for file {d}')

            # establish run order from acquisition data (in _INIT/_HEADER file)

    def plotFun(self, id='4', iid=0, **xargs):
        {k: x.name for k, x in self.eres[0].efun.funcs.items()}
        self.eres[0].plotEfun('26', prominence=10000, height=800, width=5)
        # , prominence=10000, height=800, width=5
        import plotly.graph_objects as go

        fig = go.Figure()

        areas = {}
        for rec in self.eres:
            areas[rec.fname]={}
            for f in rec.efun.funcs: # for every function
                print(f)
                id = f.replace('FUNCTION ', '')
                print(id)
                try:
                    efDat = rec.extractFunData(id)
                    for i, fd in enumerate(efDat['d']):
                        print(i)
                        xd = fd
                        try:
                            areas[rec.fname].update({f+'_'+str(i):  pbound(xd, plot=False)})
                        except:
                            pass
                except:
                    pass


        # mapping of functions
            map = {'a1': {'std': {'FUNCTION 5': 'Picolinic acid-D3'}, 'analyte': {'FUNCTION 4': 'Picolinic/nicotinic acid/Nicolinic acid'}},
                   'a2': {'std': {'FUNCTION 6': 'Nicotinic acid-D4'}, 'analyte': {'FUNCTION 4': 'Picolinic/nicotinic acid/Nicolinic acid'}},
                   'a3': {'std': {'FUNCTION 10': '3-HAA-D3'}, 'analyte': {'FUNCTION 8': '3-HAA'}},
                   'a4': {'std': {'FUNCTION 11': 'Dopamine-D4'}, 'analyte': {'FUNCTION 9': 'Dopamine'}},
                   'a5': {'std': {'FUNCTION 20': 'Serotonin-d4'}, 'analyte': {'FUNCTION 12': 'Serotonin'}},
                   'a6': {'std': {'FUNCTION 14': 'Tryptamine-d4'}, 'analyte': {'FUNCTION 13': 'Tryptamine'}},
                   'a7': {'std': {'FUNCTION 17': 'Quinolinic acid-D3'}, 'analyte': {'FUNCTION 15': 'Quinolinic acid'}},
                   'a8': {'std': {'FUNCTION 19': 'I-3-AA-D4'}, 'analyte': {'FUNCTION 18': 'I-3-AA'}},
                   'a9': {'std': {'FUNCTION 27': 'Kynurenic acid-D5'}, 'analyte': {'FUNCTION 22': 'Kynurenic acid'}},
                   'a10': {'std': {'FUNCTION 28': '5-HIAA-D5'}, 'analyte': {'FUNCTION 26': '5-HIAA'}},
                   'a11': {'std': {'FUNCTION 35': 'Tryptophan-d5'}, 'analyte': {'FUNCTION 30': 'Tryptophan'}},
                   'a12': {'std': {'FUNCTION 33': 'Xanthurenic acid-D4'}, 'analyte': {'FUNCTION 31': 'Xanthurenic acid'}},
                   'a13': {'std': {'FUNCTION 36': 'Kynurenine-D4'}, 'analyte': {'FUNCTION 32': 'Kynurenine'}},
                   'a14': {'std': {'FUNCTION 39': '3-HK 13C15N'}, 'analyte': {'FUNCTION 38': '3-HK'}},
                   'a15': {'std': {'FUNCTION 41': 'Neopterin-13C5'}, 'analyte': {'FUNCTION 40': 'Neopterin'}},
                   'a16': {'analyte': {'FUNCTION 3': '2-aminophenol'}},
                   'a17': {'analyte': {'FUNCTION 42': '3'}},
                   'a18': {'analyte': {'FUNCTION 16': '3-methoxy-p-tyramine'}},
                   'a19': {'analyte': {'FUNCTION 34': '4-hydroxyphenylacetylglycine'}},
                   'a20': {'analyte': {'FUNCTION 23': '5-ME_TRYT'}},
                   'a21': {'analyte': {'FUNCTION 37': '5-OH-tryptophan'}},
                   'a22': {'analyte': {'FUNCTION 21': '(-)-Epinephrine'}},
                   'a23': {'analyte': {'FUNCTION 29': 'L-Dopa'}},
                   'a24': {'analyte': {'FUNCTION 25': 'L-tryptophanol'}},
                   'a25': {'analyte': {'FUNCTION 43': 'N-acetyl-L-tyrosine'}},
                   'a26': {'analyte': {'FUNCTION 24': 'N-methylseotonin'}},
                   'a27': {'analyte': {'FUNCTION 1': 'Trimethylamine'}},
                   'a28': {'analyte': {'FUNCTION 2': 'Trimethylamine-N-oxide'}},
                   'a29': {'analyte': {'FUNCTION 7': 'Tyramine'}},
                   }

        sr = Trp.waters(rawf, efun=efun)
        sr.plotEfun(id=4, prominence=[None, 0.2], height=0.1, width=1)

        for i in range(43):
            print(i + 1)
            sr.plotEfun(id=6, prominence=10000, height=800, width=5)

        # generate calibration curves for 15 compounds where possible
        for k, d in map.items():
            # get number of SIR
            fid = list(d['analyte'].keys())[0]
            dfsub=df.iloc[:,df.columns.str.contains(fid+'_')]






    df = pd.DataFrame(areas).T
        plt.close('all')
        tt=df['FUNCTION 26_1'] / df['FUNCTION 28_1']
        plt.scatter(np.arange(df.shape[0]), tt)
        exp=df.index.str.split('_').str[6].str.replace('Cal', '').astype(int)
        plt.scatter(exp, np.log(tt+1))
        plt.scatter(exp, tt)





            efDat = [x.extractFunData(id) for x in self.eres]

            for k, e in enumerate(dd):
                print(k)
                # sub = e['m']
                xrawd = e['d']
                xd = xrawd[1]

        dd = []
        for x in self.eres:
            # if '2_Cal' in x.fname:
            #     print(x.fname)
            #     dd.append(x.extractFunData(id))
            dd.append(x.extractFunData(id))

        nReact = len(dd[0])
        # # fig, axs = plt.subplots(nReact, sharex=True, sharey=True)
        # axs[0].set_ylabel(r"$\bfCount$")
        # axs[1].set_ylabel(r"$\bfCount$")

        id='26'
        areas = {id: []}
        for k, e in enumerate(dd):
            print(k)
            # sub = e['m']
            xrawd =e['d']
            xd = xrawd[1]

            areas[id].append(pbound(xd, plot=False))
            if k > 10:
                break

        plt.scatter(np.arange(len(areas)), areas)


            for i in iid:
                xd = xrawd[i]
                fig.add_trace(go.Scatter(
                    x = xd[1],
                    y =xd[2],  name=e['m'][i]['fname']),
               )

        fig.update_layout(title=x.efun.funcs['FUNCTION '+id].name,)
        fig.show()

import scipy.signal as ss
import scipy.stats as st


def pbound(xd, plot=True):
    # normalisation
    imax = max(xd[2])
    if imax > 100000:
        yn = xd[2] /imax
        # smoothing
        ysm = ss.savgol_filter(yn, 21, 3)

        # find local maxima/minima
        peaks, heights = ss.find_peaks(ysm, 0.1)
        w, wh, lh, rh = ss.peak_widths(ysm, peaks)
        npeaks = ss.find_peaks(-ysm)[0]

        dd = (npeaks - peaks[0])
        iid = [peaks[0] + min([x for x in dd if x > 0]), peaks[0] - abs(max([x for x in dd if x < 0]))]

        # max likelihood normal: regression with
        ym = st.norm.pdf(xd[1], loc=xd[1][peaks[0]], scale=w[0] / 540)
        ym = ym / max(ym) * max(yn)

        idxEval = [i for i, x in enumerate(zip((xd[1] < xd[1][iid[0]]) & (xd[1] > xd[1][iid[1]]))) if x[0]]

        xint = [xd[1][i] for i in idxEval]
        yint = [xd[2][i] for i in idxEval]

        a = np.trapz(yint, xint)

        if plot:
            plt.figure()
            plt.plot(xd[1], yn*imax)
            plt.title('Area = '+str(int(a)))
            plt.plot(xd[1], ysm*imax)
            # plt.scatter(xd[1][iid], yn[iid]*imax)
            plt.fill_between(
                x=xd[1],
                y1=yn*imax,
                where=(xd[1] < xd[1][iid[0]]) & (xd[1] > xd[1][iid[1]]),
                color="b",
                alpha=0.2)
            plt.fill_between(
                x=xd[1],
                y1=ysm * imax,
                where=(xd[1] < xd[1][iid[0]]) & (xd[1] > xd[1][iid[1]]),
                color="b",
                alpha=0.2)
            plt.plot(xd[1], ym*imax)
    else:
        a = 0
        if plot:
            plt.plot(xd[1], xd[2])

    return a

























# plt.scatter(xd[1], xd[2])
plt.plot(xd[1], xd[2])

peaks, heights = ss.find_peaks(xd[2], height=100000)
w, wh, lh, rh = ss.peak_widths(xd[2], peaks)

xd[1][int(peaks[0])]-(w*0.5)

stepsize = np.mean(np.diff(xd[1]))
rb = xd[1][peaks[0]]-(w*0.5*stepsize)
lb = xd[1][peaks[0]]+(w*0.5*stepsize)

plt.vlines(rb, 1, 1000000)
plt.vlines(lb, 1, 1000000)



# maximum likelihood of Gaussian
import numpy.random as nr
ss.gaussian(len(xd[0]), std=(w*0.5*stepsize)[0])

ym = st.norm.pdf(xd[1], loc=xd[1][peaks[0]], scale=w/540)
ymn = ym/max(ym)*max(xd[2])
plt.plot(xd[1], ymn)
plt.fill_between(
    x=xd[1],
    y1=ymn,
    # where=(-1 < t) & (t < 1),
    color="b",
    alpha=0.2)

# second deriv info
import numpy as np
plt.plot(xd[1], np.gradient(xd[2]))
plt.plot(xd[1], np.gradient(np.gradient(xd[2])))



# smooth
yn = xd[2] / max(xd[2])
plt.plot(xd[1], yn)
ysm = ss.savgol_filter(yn, 21, 3)
plt.plot(xd[1], ysm)
peaks, heights = ss.find_peaks(ysm, 0.1)
plt.scatter(xd[1][peaks], heights['peak_heights'])

npeaks = ss.find_peaks(-ysm)[0]

npeaks
dd = (npeaks - peaks[0])
iid = [peaks[0] + min([x for x in dd if x > 0]), peaks[0] - abs(max([x for x in dd if x < 0]))]


plt.scatter(xd[1][iid], yn[iid])


def pbound(xd, plot=True):
    # normalisation
    imax = max(xd[2])
    yn = xd[2] /imax
    # smoothing
    ysm = ss.savgol_filter(yn, 21, 3)

    # find local maxima/minima
    peaks, heights = ss.find_peaks(ysm, 0.1)
    w, wh, lh, rh = ss.peak_widths(xd[2], peaks)
    npeaks = ss.find_peaks(-ysm)[0]

    dd = (npeaks - peaks[0])
    iid = [peaks[0] + min([x for x in dd if x > 0]), peaks[0] - abs(max([x for x in dd if x < 0]))]

    # max likelihood normal: regression with
    ym = st.norm.pdf(xd[1], loc=xd[1][peaks[0]], scale=w[0] / 540)
    ym = ym / max(ym) * max(yn)

    iid = [i for i, x in enumerate(zip((xd[1] < xd[1][iid[0]]) & (xd[1] > xd[1][iid[1]]))) if x[0]]

    xint = [xd[1][i] for i in iid]
    yint = [xd[2][i] for i in iid]

    a = np.trapz(yint, xint)

    if plot:
        plt.plot(xd[1], yn*imax)
        plt.title('Area = '+str(int(a)))
        # plt.plot(xd[1], ysm)
        # plt.scatter(xd[1][iid], yn[iid]*imax)
        plt.fill_between(
            x=xd[1],
            y1=yn*imax,
            where=(xd[1] < xd[1][iid[0]]) & (xd[1] > xd[1][iid[1]]),
            color="b",
            alpha=0.2)
        plt.fill_between(
            x=xd[1],
            y1=ysm * imax,
            where=(xd[1] < xd[1][iid[0]]) & (xd[1] > xd[1][iid[1]]),
            color="b",
            alpha=0.2)
        plt.plot(xd[1], ym*imax)

    return a

xd[1][iid[1]]
np.array(dd)
dd

dd[dd>0]]

npeaks, heights = ss.find_peaks(-ysm)
plt.scatter(peaks, ysm[peaks])

ysm[:-1] - ysm[1: ]

rpeaks, rheights = ss.find_peaks(-ysm)




    # smooth and find minimum values next to maximum




#
        #
        #
        #
        #         axs[i].plot(xd[1], xd[2])
        #         # pidx, pint = self.eres[0].quant(xd, **xargs)
        #         pidx, pint = self.eres[0].quant(xd, prominence=10000, height=100000, width=5)
        #         axs[i].scatter(xd[1][pidx], pint['peak_heights'], c='red')
        #         if k ==0:
        #             ff = self.eres[k].efun.funcs['FUNCTION ' + id]
        #             fig.suptitle(f"F{str(id)}:  {ff.name} @ {ff.reactions[str(i + 1)].massPrecursor} m/z")
        #             axs[i].text(0.77, 0.9, f'Pr@ {ff.reactions[str(i + 1)].massProduct} m/z', rotation=0, fontsize=8,
        #                         transform=axs[i].transAxes,
        #                         ha='left')
        #             axs[i].text(0.77, 0.8, f'st:  {ff.stEnd_min}-{ff.stStart_min} min', rotation=0, fontsize=8,
        #                         transform=axs[i].transAxes,
        #                         ha='left')
        #             axs[i].text(0.77, 0.7, f'{ff.reactions[str(i + 1)].compoundName}', rotation=0, fontsize=8,
        #                         transform=axs[i].transAxes,
        #                         ha='left')
        #             axs[i].text(0.77, 0.5, f'Co-E: {ff.reactions[str(i + 1)].collisionEnergy} eV', rotation=0,
        #                         fontsize=8,
        #                         transform=axs[i].transAxes,
        #                         ha='left')
        #             axs[i].text(0.77, 0.4, f'Conevolt: {ff.reactions[str(i + 1)].conevoltageV}', rotation=0, fontsize=8,
        #                         transform=axs[i].transAxes,
        #                         ha='left')
        #             axs[i].text(0.77, 0.3, f'Polarity: {ff.polarity}', rotation=0, fontsize=8,
        #                         transform=axs[i].transAxes,
        #                         ha='left')
        #             axs[i].text(0.77, 0.2, f'P@ {ff.reactions[str(i + 1)].massPrecursor} m/z', rotation=0, fontsize=8,
        #                         transform=axs[i].transAxes,
        #                         ha='left')
        #     axs[i].set_xlabel(r"$\bfScantime (min)$")
        #



test = ExpSet(path='/Volumes/ANPC_ext1/trp_qc/')
test.plotFun(id='10')
self=test
for i in range(5):
    print(i+1)
    test.eres[i].plotEfun(id=2, prominence=10000, height=800, width=5)
    sr.plotEfun(id=i+1, prominence=10000, height=800, width=5)



test = ReadW(rawf, convert=True)
test = ReadW(rawf, convert=False)
test.edf
df=self.dfd['1P']


ipath='/Volumes/ANPC_ext1/trp_qc/RCY_TRP_023_Robtot_Spiking_04Aug2022_URN_LTR_87.raw/_INLET.INF'
ipath='/Volumes/ANPC_ext1/trp_qc/RCY_TRP_023_Robtot_Spiking_04Aug2022_URN_LTR_87.raw/_HEADER.TXT'


epath='/Users/TKimhofer/Downloads/Torben19Aug.exp'



x=SIRFunction(pars['FUNCTION 2'])

ff=[SIRFunction(x) for k,x in pars.items() if 'FUN' in k]
for k,x in pars:i
    print(k)


import matplotlib.pyplot as plt
X=exp.xrawd['1N']
import numpy as np

np.unique(X[0])
idx=np.where(X[0]==90.)[0]
plt.plot(X[1][idx], X[2][idx])

exp.xrawd['1P']
ttt=exp.dfd['1N']
len(np.unique(exp.xrawd['1P'][0]))



test=pd.DataFrame(exp.dfd['1P'].id.apply(srmSic))



    @typechecked
    class ReadB:
        """Bruker experiment class and import methods"""

        def __init__(self, dpath: str, convert: bool = True,
                     docker: dict = {'repo': 'chambm/pwiz-skyline-i-agree-to-the-vendor-licenses:latest'}):
            """Import of XC-MS level 1 scan data from raw Bruker or msmate data format (.d or binary `.p`, respectively)

               Note that this class is designed for use with `MsExp`.

               Args:
                   dpath: Path `string` variable pointing to Bruker experiment folder (.d)
                   convert: `Bool` indicating if conversion should take place using docker container inf `.p` is not present
                   docker: `dict` with docker information: `{'repo': 'convo:v1', 'mslevel': 0, 'smode': [0, 1, 2, 3, 4, 5], 'amode': 2, 'seg': [0, 1, 2, 3], }`
            """


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
            ec = f'docker run -it --rm -e WINEDEBUG-all -v "{os.path.dirname(self.dpath)}":/data {self.docker["repo"]} chambm/pwiz-skyline-i-agree-to-the-vendor-licenses wine msconvert /data/{self.fname}'
            print(ec)
            os.system(ec)
            t1 = time.time()
            print(f'Conversion time: {np.round(t1 - t0)} sec')

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
                [(polmap[x['Polarity']], str(x['MzAcqRangeLower']) + '-' + str(x['MzAcqRangeUpper']), x['MsLevel']) for
                 x in
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


class TrpExp:
    """mzML experiment class and import methods"""
    def __init__(self, fpath: str, mslev:str='targeted'):
        """Import of XC-MS level 1 scan data from mzML files V1.1.0

            Note that this class is designed for use with `MsExp`.

            Args:
                fpath: Path `string` variable pointing to mzML file
                mslev: `String` defining ms level read-in (fulls can)
        """
        self.fpath = fpath
        if mslev != 'targeted':
            raise ValueError('Check mslev argument - This function currently support ms level 1 import only.')
        self.mslevel = mslev
        self._read_mzml()
        self._createSpectMat()
        self._stime_conv()
        # self.df['subsets'] = self.df.ms_level+self.df.scan_polarity
        # ids = np.unique(self.df.subsets, return_counts=True)
        # imax = np.argmax(ids[1])
        # self.ms0string = ids[0][imax]
        # self.ms1string = None
        # if len(ids[0]) > 1:
        #     print(f'Experiment done with polarity switching! Selecting {ids[0][imax]} as default level 1 data set. Alter attribute `ms0string` to change polarity')
        #     self.xrawd = {}
        #     self.dfd = {}
        #     print(self.Xraw.shape)
        #     Xr = pd.DataFrame(self.Xraw.T)
        #     Xr.index=self.Xraw[0]
        #     for i, d in enumerate(ids[0]):
        #         self.dfd.update({d: self.df[self.df.subsets == d]})
        #         self.xrawd.update({d: Xr.loc[self.dfd[d]['index'].astype(int).values].to_numpy().T})
        # else:
        #     self.xrawd = {self.ms0string: self.Xraw}
        #     self.dfd = {self.ms0string: self.df}

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

    def prep_df(self, df):
        if 'MS:1000127' in df.columns:
            add = np.repeat('centroided', df.shape[0])
            add[~(df['MS:1000127'] == True)] = 'profile'
            df['MS:1000127'] = add
        if 'MS:1000128' in df.columns:
            add = np.repeat('profile', df.shape[0])
            add[~(df['MS:1000128'] == True)] = 'centroided'
            df['MS:1000128'] = add

        df = df.rename(
            columns={'UO:0000010': 'time_unit', 'UO:0000031': 'time_unit', 'defaultArrayLength': 'n',
                     'MS:1000129': 'polNeg', 'MS:1000130': 'polPos',
                     'MS:1000127': 'specRepresentation', 'MS:1000128': 'specRepresentation',
                     'MS:1000505': 'MaxIntensity', 'MS:1000285': 'SumIntensity',
                     'MS:1000016': 'Rt'
                     })
        df.columns = [self.obo_ids[x]['name'].replace(' ', '_') if x in self.obo_ids.keys() else x for x in
                           df.columns.values]
        df['fname'] = self.fpath
        return df

    def _createSpectMat(self):
        # tyr targeted data has two dimensions: 1. rt, 2. intensity
        """Organise raw MS data and scan metadata"""
        sc_msl1 = [(i, len(x['data']['Int']['d'])) for i, x in self.out31.items() if len(x['data']) == 2] # sid of ms level 1 scans

        self.df = pd.DataFrame([x['meta'] for i, x in self.out31.items() if len(x['data']) == 2])
        self.df['defaultArrayLength'] = self.df['defaultArrayLength'].astype(int)

        from collections import defaultdict, Counter
        xrawd = {}
        if 'MS:1000129' in self.df.columns:
            nd = self.df['defaultArrayLength'][self.df['MS:1000129'] == True].sum()
            xrawd['1N'] = np.zeros((4, nd))

        if 'MS:1000130' in self.df.columns:
            nd = self.df['defaultArrayLength'][self.df['MS:1000130'] == True].sum()
            xrawd['1P'] = np.zeros((4, nd))

        row_counter = Counter({'1P': 0, '1N': 0})
        sid_counter = Counter({'1P': 0, '1N': 0})
        dfd = defaultdict(list)
        for i, s in enumerate(sc_msl1):

            d = self.out31[s[0]]
            if s[1] != self.df['defaultArrayLength'].iloc[i]:
                raise ValueError('Check data extraction')

            cbn = [[d['meta']['index']] * s[1]]
            for k in d['data']:
                cbn.append(d['data'][k]['d'])

            if 'MS:1000129' in d['meta']:
                fstr = '1N'
            elif 'MS:1000130' in d['meta']:
                fstr = '1P'

            cbn.append([sid_counter[fstr]] * s[1])
            add = np.array(cbn)
            xrawd[fstr][:, row_counter[fstr]:(row_counter[fstr] + s[1])] = add

            dfd[fstr].append(d['meta'])

            sid_counter[fstr] += 1
            row_counter[fstr] += (s[1])

        for k, d, in dfd.items():
            print(k)
            dfd[k] = self.prep_df(pd.DataFrame(dfd[k]))

        self.dfd = dfd
        self.xrawd = xrawd
        self.ms0string = row_counter.most_common(1)[0][0]
        self.ms1string = None

    def _stime_conv(self):
        pass
        """Performs scan time conversion from minutes to seconds."""
        for k, d, in self.dfd.items():
            if 'min' in d['time_unit'].iloc[1]:
                self.dfd[k]['Rt'] = d['Rt'].astype(float) * 60
                self.dfd[k]['time_unit'] = 'sec'
                self.xrawd[k][3] = self.xrawd[k][3] * 60



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

        Perform feature detection using dbscan and signal qc filters.

        Features are detected in sID/mz dimensions after scaling the mz dimension and classified into four different quality categories:
            L0 - All features (unfiltered)
            L1 - Features comprising of n or more data points (`st_len_min`)
            L2 - Features that do not pass one or more of the defined quality filters (see below)
            L3 - Features that pass all quality filters (see below)

        Quality filter:

            ppm: Variability of a feature in m/z dimension (ppm)
            sID_gap: Fraction of consecutive scan IDs of data points representing a feature (`float` {0-1} with zero meaning no gaps/doublets allowed)
            non_neg: Fraction of intensity values falling below zero after baseline correction (`float` {0-1} with zero meaning no values below zero intensity)
            sino: Minimum signal-to-noise ratio (noise determined locally)
            raggedness: Quantifies the degree of intensity raggeness of a feature (`float` {0-1} with zero meaning monotonic in/decrease before/after peak maximum)

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

    def _cls(self, cl_id): #Xf, qc_par
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
        idx = np.where(self.Xf[7] == cl_id)[0]
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

        if ppm > self.qc_par['ppm']:
            crit = 'm/z variability of a feature (ppm)'
            od = {'flev': 2, 'crit': crit, 'qc': qc, 'quant': quant, 'descr': descr, 'fdata': fdata}
            return od

        # gaps or doublets in successive scan ids
        sid = self.Xf[4, idx]
        fdata.update({'sid': sid})
        sid_diff = np.diff(sid)
        sid_gap = (np.sum(sid_diff[sid_diff > 1])) / (iLen - 1) #+ np.sum(np.abs(sid_diff[(sid_diff < 1)]))
        qc['sId_gap'] = sid_gap

        if sid_gap > self.qc_par['sId_gap']:
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

        bcor_argmax = np.argmax(ybcor)
        bcor_max = np.max(ybcor * ysm_max)

        descr.update({'mzMaxI': mz[bcor_argmax], 'rtMaxI': st[bcor_argmax]})

        fdata.update({'I_sm_bline': ybcor * ysm_max, 'I_smooth': ysm * ysm_max, 'mz': mz})
        nneg = np.sum(ybcor >= (-0.05)) / len(ysm)
        qc['non_neg'] = nneg
        if nneg < self.qc_par['non_neg']:
            crit = 'nneg: Neg Intensity values'
            od = {'flev': 2, 'crit': crit, 'qc': qc, 'quant': quant, 'descr': descr, 'fdata': fdata}
            return od
        # signal / noise
        scans = np.unique(sid)
        scan_id = np.concatenate([np.where((self.Xf[4] == x) & (self.Xf[5] == 1))[0] for x in scans])
        noiI = np.median(self.Xf[2, scan_id]) if len(scan_id) >0 else np.quantile(self.Xf, 0.8)
        sino = bcor_max / noiI
        qc['sino'] = sino
        if sino < self.qc_par['sino']:
            crit = 'sino: s/n below qc threshold'
            od = {'flev': 2, 'crit': crit, 'qc': qc, 'quant': quant, 'descr': descr, 'fdata': fdata}
            return od
        # intensity variation level
        idx_maxI = np.argmax(ybcor)
        sdxdy = (np.sum(np.diff(ybcor[:(idx_maxI+1)]) < (-0.01)) + np.sum(np.diff(ybcor[(idx_maxI):]) > 0.01)) / (
                len(ybcor) - 1) # this is zero if not ragged (np.diff flips array)
        qc['raggedness'] = sdxdy # this is zero if not ragged

        if qc['raggedness'] > self.qc_par['raggedness']:
            crit = 'raggedness: intensity variation level above qc threshold'
            od = {'flev': 2, 'crit': crit, 'qc': qc, 'quant': quant, 'descr': descr, 'fdata': fdata}
            return od
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
        print(
            f'Number of L1 features: {len(cl_n_above)} ({np.round(len(cl_n_above) / len(self.cl_labels[0]) * 100, 1)}%)')
        self.feat.update(
            {'id:' + str(x): {'flev': 1, 'crit': 'n < st_len_min'} for x in self.cl_labels[0][~min_le_bool]})
        t2 = time.time()
        c = 0
        for fid in cl_n_above:
            c += 1
            # print(fid)
            f1 = self._cls(cl_id=fid) #, Xf=self.Xf, qc_par=self.qc_par
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

        print(
            f'Time for {len(cl_n_above)} feature summaries: {np.round(t3 - t2)}s ({np.round((t3 - t2) / len(cl_n_above), 2)}s per ft)')

        if (len(self.feat_l2) == 0) & (len(self.feat_l3) == 0):
            raise ValueError('No L2 features found')
        else:
            print(
                f'Number of L2 features: {len(self.feat_l2) + len(self.feat_l3)} ({np.round((len(self.feat_l2) + len(self.feat_l3)) / len(cl_n_above) * 100)}%)')

        if len(self.feat_l3) == 0:
            # raise ValueError('No L3 features found')
            print('No L3 features found')
        else:
            self._l3toDf()
            print(
                f'Number of L3 features: {len(self.feat_l3)} ({np.round(len(self.feat_l3) / (len(self.feat_l2) + len(self.feat_l3)) * 100, 1)}%)')

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



@typechecked
class Viz:
    """ Base class defining visualisation modalities for MS data.

    Note that this class is designed for use with `MsExp`.
    """

    def chromg(self, tmin: float = None, tmax: float = None, ctype: list = ['tic', 'bpc'], xic_mz: float = [],
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

        # @log

    def vizpp(self, selection: Union[dict, None] = None, lev: int = 3):
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
        axs[1].set_xlabel(r"$\bfScan time$, s")
        axs[1].yaxis.offsetText.set_visible(False)
        axs[1].yaxis.set_label_text(r"$\bfm/z$")

    def vizIso(self, selection: Union[dict, None] = None):
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
        print(i)
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


@typechecked
class MsExp(MSstat):
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
    def waters(cls, fpath: str, mslev: str = '1'):
        # cls.__name__ = 'mzml from file import'
        da = ReadW(fpath, mslev)
        return cls(dpath=da.fpath, fname=da.fpath, mslevel=da.mslevel, ms0string=da.ms0string, ms1string=da.ms1string,
                   xrawd=da.xrawd, dfd=da.dfd, summary=False)

    @classmethod
    def rM(cls, da: ReadM):
        # cls.__name__ = 'data import from msmate ReadM'
        return cls(dpath=da.fpath, fname=da.fpath, mslevel=da.mslevel, ms0string=da.ms0string, ms1string=da.ms1string,
                   xrawd=da.xrawd, dfd=da.dfd, summary=False)

    @classmethod
    def rW(cls, da: ReadM):
        # cls.__name__ = 'data import from msmate ReadM'
        return cls(dpath=da.fpath, fname=da.fpath, mslevel=da.mslevel, ms0string=da.ms0string, ms1string=da.ms1string,
                   xrawd=da.xrawd, dfd=da.dfd, summary=False)

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

sys.path.append('/Users/tkimhofer/pyt/pyms')
import msmate as ms
fpath='/Volumes/ANPC_ext1/trp_qc/NW_TRP_023_manual_Spiking_05Aug2022_Cal2_138.mzML'
r = ms.MsExp.mzml('/Volumes/ANPC_ext1/trp_qc/NW_TRP_023_manual_Spiking_05Aug2022_Cal2_138.mzML',  mslev='1')



















