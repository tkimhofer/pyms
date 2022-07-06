import numpy as np
import requests
import xml.etree.ElementTree as ET
import re
import pandas as pd
dori = pd.read_csv('/Users/TKimhofer/Downloads/maAA - Sheet1.csv')

out=chebiRequest(chebiNb='37024')

derivA = 170.04799
monomass = float(out['monoisotopicMass'])

mm = derivA + monomass
print(out['chebiAsciiName'])
print(mm)

dori['ids']=dori.chebi.str.split(':').str[1]

res = []
for i in dori.ids:
    print(i)
    res.append(chebiRequest(i))

out=pd.DataFrame(res)

ds=pd.concat((dori, out), axis=1)
ds['dMass'] = ds['Acurate mass after derivatisation'].astype(float)-ds['monoisotopicMass'].astype(float)

fpath ='/Volumes/ANPC_ext1/ms/covid_cambridge_MS_AA_PAI03_PLASMA4B_070920_QC 2_16.d'
fpath = '/Volumes/ANPC_ext1/ms/covid_cambridge_MS_AA_PAI03_PLASMA2B_070920_QC 1_2.d'

fpath='/Volumes/ANPC_ext1/ms/covid_cambridge_MS_AA_PAI03_PLASMA4B_070920_Cal 9_5.d'

fpath='/Users/TKimhofer/Desktop/RPN_Pl/covid_cambridge_MS_RP_NEG_PAI04_PLASMA1B_LTR_11_COND2.d'
tt = msExp1(fpath)

fname='/Volumes/ANPC_ext1/ms/covid_cambridge_MS_AA_PAI03_PLASMA4B_070920_Cal 9_5.d'
type='run'
mslevel=0
smode=0
amode=2
seg=3


class Exp1:
    def __init__(self, fname, type, mslevel, smode, amode, seg, verbosity=0):
        import sqlite3, os
        self.fname = fname
        self.type = type
        self.mslevel = mslevel
        self.smode = smode if not isinstance(smode, list) else tuple(smode)
        self.amode = amode
        self.seg = seg if not isinstance(seg, list) else tuple(seg)
        sqlite_fn = '/Volumes/ANPC_ext1/ms/covid_cambridge_MS_AA_PAI03_PLASMA4B_070920_Cal 9_5.d/analysis.sqlite'
        self.con = sqlite3.connect(sqlite_fn)
        self.con.row_factory = sqlite3.Row

        self.collectProp()
        self.collectMeta()

    def collectProp(self):
        q=self.con.execute(f"select * from Properties")
        res = [dict(x) for x in q.fetchall()]
        prop = {}; [prop.update({x['Key']: x['Value']}) for x in res]
        self.properties = prop

    def collectMeta(self):
        q = self.con.execute(f"select * from SupportedVariables")
        vlist = [dict(x) for x in q.fetchall()]
        vdict = {x['Variable']: x for x in vlist}
        qst = "SELECT * FROM Spectra as s JOIN AcquisitionKeys as ak ON s.AcquisitionKey=ak.Id ORDER BY Rt"
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


fname='/Volumes/ANPC_ext1/ms/covid_cambridge_MS_AA_PAI03_PLASMA4B_070920_Cal 9_5.d'
sqlite_fn = '/Volumes/ANPC_ext1/ms/covid_cambridge_MS_AA_PAI03_PLASMA4B_070920_Cal 9_5.d/analysis.sqlite'
self=Exp1(fname, 'meta', mslevel, smode, amode, seg)


self.edat


self.seg=3
self.smode=(0, 3)
f"mm8v3_sm{self.smode}"

'='if not isinstance(self.seg, tuple) else 'in'








fpath='/Users/TKimhofer/Desktop/RPN_Pl/covid_cambridge_MS_RP_NEG_PAI03_PLASMA5B_170920_COV00689_CV0215_73_491_NEG_491.d'

i=35
print(ds['Unnamed: 0'].iloc[i])
print(ds.definition.iloc[i])
mzi = ds['Extracted mass'].astype(float).iloc[i]
rti = ds['Et'].astype(float).iloc[i]*60
print(ds['Acurate mass after derivatisation'].iloc[i])
print(mzi)
print(rti)

fpath='/Volumes/ANPC_ext1/ms/covid_cambridge_MS_AA_PAI03_PLASMA5B_090920_Cal 1_13.d'
fpath = '/Users/TKimhofer/Desktop/barwin20_C1_MS_AA_PAI04_BARWINp01_250221_QC 2_17.d'

fpath1 = '/Users/TKimhofer/Desktop/AA BARWON/barwin20_C1_MS_AA_PAI04_BARWINp01_250221_QC 2_17.d'
tt1 = MSexp(fpath1)

fpath='/Users/TKimhofer/Desktop/RPN_Pl/covid_cambridge_MS_RP_NEG_PAI03_PLASMA5B_170920_COV00689_CV0215_73_491_NEG_491.d'
fpath='/Users/TKimhofer/Desktop/RPN_Pl/other/covid_cambridge_MS_AA_PAI04_PLASMARR_140920_COV648_CV0080_v2.d'
fpath='/Users/TKimhofer/Desktop/RPN_Pl/other/covid_cambridge_MS_AA_PAI04_PLASMARR_140920_COV00733_CV0209_24_v2 .d'
fpath='/Users/TKimhofer/Desktop/RPN_Pl/other/covid_cambridge_MS_AA_PAI05_PLASMA8B_140920_QC 1_38.d'
fpath='/Users/TKimhofer/Desktop/RPN_Pl/other/covid_cambridge_MS_AA_PAI05_PLASMA8B_140920_QC 2_47.d'
fpath='/Users/TKimhofer/Desktop/RPN_Pl/other/covvac_C3_PLA_MS-AA_PAI03_VAXp41_280322_QC01_2.d'
fpath='/Users/TKimhofer/Desktop/RPN_Pl/other/covvac_C3_PLA_MS-AA_PAI03_VAXp41_280322_QC01_16.d'
fpath='/Users/TKimhofer/Desktop/RPN_Pl/other/covvac_C3_PLA_MS-AA_PAI03_VAXp41_280322_QC01_43.d'
fpath='/Users/TKimhofer/Desktop/RPN_Pl/other/covvac_C3_PLA_MS-AA_PAI03_VAXp41_280322_QC02_15.d'
fpath='/Users/TKimhofer/Desktop/RPN_Pl/other/covvac_C3_PLA_MS-AA_PAI03_VAXp41_280322_QC02_83.d'


fpath='/Volumes/Backup Plus/ms_apple_sc/BravoFlour-methanol_dda_1-31-2022_1-B,2_4713.d'
fpath='/Volumes/alims4ms/PAI01/Joel/Method Development_Polar/2020 0319_PAI1_AMIDE_Extn solvents test/2020_0319_PAI01_AMIDE Cont_MEOH_FA0_2-A,12_1_584.d'
fpath='/Users/TKimhofer/Desktop/RPN_Pl/other/covid_cambridge_MS_AA_PAI04_PLASMARR_140920_COV00733_CV0209_24_v2 .d'
self = ExpSum(fpath)
ds= self.ds

self.properties
self=MSexp(fpath)


fhilic='/Users/TKimhofer/Desktop/RPN_Pl/other/6945.d'
os = ExpSum(fhilic)
print(o.loc['Segment 3'])
o.summary.loc['Segment 3'].values
o = MSexp(fhilic, docker={'repo': 'convo:v1', 'mslevel': 0, 'smode': [0, 1, 2, 3 ,4, 5], 'amode':2 , 'seg': [1, 2, 3],})
o.plt_chromatogram(ctype=['tic', 'bpc'])


('PAI01', '1825265.10252', 'impact II')
('PAI03', '1825265.10251', 'impact II')
('PAI04', '1825265.10274', 'impact II')
('PAI05', '1825265.10271', 'impact II')
('PAI06', '1825265.10258', 'impact II')
('PAI07', '1825265.10267', 'impact II')
('PAI08', '1825265.10269', 'impact II')

('PAT01', None, 'Waters')
('PAT02', None, 'Waters')
('PLIP01', None, 'Sciex')
('RAI01', None, 'Waters')

('RAI02', '1825265.10275', 'impact II')
('RAI03', '1825265.10273', 'impact II')
('RAI04', '1825265.10272', 'impact II')
('RAT01', None, 'Waters')
('REIMS01', None, 'Waters')
('TIMS01', None, 'Bruker') # no sqlite3
('TIMS02', None, 'Bruker') # no sqlite3
('EVOQ', None, 'Bruker TQ') # no sqlite3



o = ExpSum(fpath)
print(o.loc['Segment 3'])
ds= o.ds
o.summary.loc['Segment 3'].values
o = MSexp(fpath, docker={'repo': 'convo:v1', 'mslevel': 0, 'smode': [0, 1, 2, 3 ,4, 5], 'amode':2 , 'seg': [1, 2, 3],})
o.plt_chromatogram(ctype=['tic', 'bpc'])

fpath='/Volumes/Backup Plus/ms_apple_sc/BravoFlour-30ace_BBCID_1-31-2022_1-F,2_4700.d'
# dpath=fpath
o = ExpSum(fpath)
o.summary.loc['Segment 3'].values
o1 = MSexp(fpath, docker={'repo': 'convo:v1', 'mslevel': 0, 'smode': [0, 1, 2, 3 ,4, 5], 'amode':2 , 'seg': [1, 2, 3],})
# o2 = MSexp(fpath, docker={'repo': 'convo:v1', 'mslevel': 0, 'smode': [0, 1, 2, 3 ,4, 5], 'amode':2 , 'seg': [1, 2, 3],})
o1.plt_chromatogram(ctype=['tic', 'bpc'])
o1.summary['Segment 3']

s = MSexp(fpath, docker={'repo': 'convo:v1', 'mslevel': 0, 'smode': [0, 1, 2, 3 ,4, 5], 'amode':2 , 'seg': [3],})
s.summary.T
s.calcSignalWindow()
np.max(s.Xsignal[...,3])

s.plt_chromatogram(ctype=['tic', 'bpc', 'xic'], xic_mz)

import idna


import os

fh=os.listdir('/Users/TKimhofer/Desktop/RPN_Pl/')

for i in fh:
    if 'covi' in i:
        s = MSexp('/Users/TKimhofer/Desktop/RPN_Pl/'+i)
        s.calcSignalWindow()





#selection={'mz_min': 300, 'mz_max': 320, 'rt_min': 60, 'rt_max': 120}

fpath1='/Users/TKimhofer/Desktop/RPN_Pl/covid_cambridge_MS_RP_NEG_PAI04_PLASMA1B_LTR_22_NEG_22.d'
s1 = msExp1(fpath1)
#tt1.vis_spectrum1(selection=selection, q_noise=0.59)

selection={'mz_min': 155, 'mz_max': 165, 'rt_min': 60, 'rt_max': 70}

s0={'mz_min': 0, 'mz_max': 1500, 'rt_min': 0, 'rt_max': 1e6}
s2={'mz_min': 100, 'mz_max': 300, 'rt_min': 120, 'rt_max': 300}
s3={'mz_min': 400, 'mz_max': 600, 'rt_min': 60, 'rt_max': 300}
s4={'mz_min': 280, 'mz_max': 300, 'rt_min': 20, 'rt_max': 70}
s.vis_spectrum(q_noise=0.999, selection=s4)


s=s1=tt1

# s.vis_spectrum1(q_noise=0.6, selection=selection)
s.getScaling(q_noise=0.95, selection=s3, qcm_local=True)

s.vis_spectrum1(q_noise=0.999, selection=s3, qcm_local=True)
s.vis_spectrum1(q_noise=0.99, selection=s3, qcm_local=False)


ppmEmp = (s.dpick[3]-s.dpick[1])/s.dpick[1] * 1e6
print(f'ppm: {(s.dpick[3]-s.dpick[1])/s.dpick[1] * 1e6}')

ppmEmp1 = (s1.dpick[3]-s1.dpick[1])/s1.dpick[1] * 1e6
print(f'ppm: {(s1.dpick[3]-s1.dpick[1])/s1.dpick[1] * 1e6}')



# X=s.Xraw[(s.Xraw[:,3]>s.dpick[0]) & (s.Xraw[:,3] < s.dpick[2]) & (s.Xraw[:,1] > s.dpick[1]) & (s.Xraw[:,1] < s.dpick[3])]
# X.shape
# Xr=X
#
# out=s.cl_summary_nocl(X, qc_par={'st_len_min': 5, 'raggedness': 0.15, 'non_neg': 0.8, 'sId_gap': 0.15, 'sino': 2, 'ppm': 15})
# out['qc']
# np.where(s.Xraw[:,])

sc=1/(s.dpick[3]-s.dpick[1])
s.dpick[3]*sc - s.dpick[1]*sc


sc1=1/(s1.dpick[3]-s1.dpick[1])

s01 = {'mz_min': 50, 'mz_max': 1000, 'rt_min': 20, 'rt_max': 600}
s.peak_picking(mz_adj=sc*0.8, q_noise=0.8, dbs_par={'eps': 1.1, 'min_samples': 3}, selection=s3,
                     qc_par={'st_len_min': 5, 'raggedness': 0.20, 'non_neg': 0.8, 'sId_gap': 0.05, 'sino': 1, 'ppm': 15})

s1.peak_picking(mz_adj=sc1*1.1, q_noise=0.5, dbs_par={'eps': 1.2, 'min_samples': 3}, selection=s01,
                     qc_par={'st_len_min': 5, 'raggedness': 0.20, 'non_neg': 0.8, 'sId_gap': 0.1, 'sino': 10, 'ppm': ppmEmp1})

s.vis_feature_pp(selection=s3, lev=3)
s.ecdf()

ss3 = {'mz_min': 458, 'mz_max': 465, 'rt_min': 106, 'rt_max': 120}
s.vis_feature_pp(selection=ss3, lev=3)
s.vis_feature_pp(selection=ss3, lev=2)

s.vis_feature_ppf(selection=s3, lev=3)
s.vis_spectrum1(q_noise=0.99, selection=s3)

import numpy as np
import matplotlib.pyplot as plt
x=np.arange(10)
f, axs = plt.subplots(2, 2)
axs[1,0].scatter(x, x)





s.l3toDf()
out=s.l3Df

out1=out.sort_values(['rt_maxI', 'mz_mean'])

s1.l3toDf()
out1=s1.l3Df


plt.scatter(out['rt_maxI'], out['mz_mean'], c=np.log(out['smbl']))
plt.scatter(out1['rt_maxI'], out1['mz_mean'], c='red', s=0.2)



out1=out.sort_values(['mz_mean')
out1.to_csv('ppick.csv')

import redis
r=redis.Redis()
r.ping()
import collections

import functools
def ex1(f):
    @functools.wraps(f)
    def wr(*args, **kwds):
        print('see you next tuesday')
        return f(*args, **kwds)
    return wr

@ex1
def hi(x=5):
    """Docstring"""
    print('test me')

hi()
hi.__doc__
hi.__name__


def greeting(name: str) -> str:
    return 'Hello ' + name

from typing import NewType

UserId = NewType('UserId', int)
some_id = UserId(524313)


import logging
logging.warning('Watch out!')
logging.info('I told you so')



s.vis_feature_pp(selection=s4, lev=3)
s02 = {'mz_min': 250, 'mz_max': 300, 'rt_min': 400, 'rt_max': 500}
s03 = {'mz_min': 460, 'mz_max': 470, 'rt_min': 490, 'rt_max': 515}
s.vis_feature_pp(selection=s2, lev=3)
s.vis_spectrum1(q_noise=0.79, selection=s2)
s.vis_feature_pp(selection=s2, lev=2)
s.vis_spectrum1(q_noise=0.99, selection=s4)
i=10

s4={'mz_min': 175, 'mz_max': 185, 'rt_min': 60, 'rt_max': 110}
s.vis_feature_pp(selection=s4, lev=2)
s.vis_spectrum1(q_noise=0.5, selection=s4)


print(dori['Unnamed: 0'].iloc[i])
mz_target = dori['Acurate mass after derivatisation'].iloc[i]
rt_target = dori['Et'].iloc[i]*60


out=chebiRequest(chebiNb=dori['chebi'].iloc[i])
derivA = 170.04799
monomass = float(out['monoisotopicMass'])

mm = derivA + monomass
print(out['chebiAsciiName'])
print(f'calcl mass: {mm}, given mass {mz_target}, mdelta {np.round(np.abs(mm-mz_target), 2)} m/z')
print(mz_target)


sf1 = {'mz_min': mz_target-2, 'mz_max': mz_target+5, 'rt_min': rt_target-60, 'rt_max': rt_target+60}
s.vis_feature_pp(selection=sf1, lev=3)
s.vis_feature_pp(selection=sf1, lev=2)



self=s
plt.scatter(self.Xf[:, 0], self.Xf[:, 5], c=np.log(self.Xf[:, 2]), s=1)
plt.scatter(self.Xf[:, 0], self.Xf[:, 1], c=np.log(self.Xf[:, 2]), s=1)
plt.scatter(self.Xf[:, 0], self.Xf[:, 1]*sc, c=np.log(self.Xf[:, 2]), s=1)

sc1 = 0.2/(s.dpick[3]-s.dpick[1])
print(sc1)
self.Xf[:, 5] = self.Xf[:, 1]*sc1

dbs = DBSCAN(eps=1.2, min_samples=3, algorithm='auto').fit(self.Xf[:, [0,5]])
print(np.unique(dbs.labels_, return_counts=True))

cl = np.unique(dbs.labels_, return_counts=True)
lab = dbs.labels_
cl[0][cl[1] < 5]


t_win = 10
dsid = np.mean(np.diff(np.sort(np.unique(self.Xraw[:,3])))) # average time between consecutive scans

wsize = int(np.round(t_win / dsid))

usids = np.unique(self.Xraw[:,0])
for i in




for i in cl[0][cl[1] < 5]:
    lab[lab==i] = -1

plt.scatter(self.Xf[:, 0], self.Xf[:, 5], c=lab, s=1)







fpath='/Users/TKimhofer/Desktop/camms/covid19_cambridgeFollowUp_MS_URPP_RAI04_COVp31_240221_COV04001_CV0943_14.d'
tt = msExp1(fpath)

selection={'mz_min': 110, 'mz_max': 120, 'rt_min': 60, 'rt_max': 120}
# tt.vis_spectrum(selection=selection)
tt1.vis_spectrum1(selection=selection, q_noise=0.99)


b=np.linspace(tt.Xraw[:,1].min(), tt.Xraw[:,1].max(), int(np.round((tt.Xraw[:,1].max()-tt.Xraw[:,1].min())*50)))

idxf = np.where(tt.Xraw[:,2] > np.quantile(tt.Xraw[:,2], 0.5))[0]
x_rev = tt.Xraw[idxf, 1]

d=np.digitize(x_rev, b)
cd=np.bincount(d) / len(np.unique(tt.Xraw[:,0]))


plt.scatter(b, cd, s=0.1)

f, ax = plt.subplots(1,1)
ax.scatter(b, cd, s=0.1)







for j in range(ds.shape[0]):
    mzi = ds['Extracted mass'].astype(float).iloc[j]
    rti = ds['Et'].astype(float).iloc[j] * 60
    print(mzi)
    selection={'mz_min': mzi-5, 'mz_max': mzi+10, 'rt_min': rti-30, 'rt_max': rti+30}
    tt.vis_spectrum(selection=selection)
    tt.peak_picking(mz_adj=140, q_noise=0.2,
                    dbs_par={'eps': 1.1, 'min_samples': 3},
                    selection=selection,
                    qc_par={'st_len_min': 5, 'raggedness': 0.15, 'non_neg': 0.5, 'sId_gap': 0.01, 'sino': 4, 'ppm': 15,
                            'tail': 3})
    tt.vis_feature_pp(selection=selection, lev=3)
    out=tt.l3toDf()

    fig, ax = tt.vis_spectrum(selection=selection)
    fig.suptitle(ds['Unnamed: 0'].iloc[j], horizontalalignment='left', fontsize=8)




tt.vis_feature_pp(selection=selection, lev=3)

# calc ion based on
mzi = ds['monoisotopicMass'].astype(float).iloc[i] + 170.04797
rti = ds['Et'].astype(float).iloc[i]*60
print(ds['Acurate mass after derivatisation'].iloc[i])
print(mzi)
print(rti)

selection={'mz_min': mzi-10, 'mz_max': mzi+10, 'rt_min': rti-30, 'rt_max': rti+30}
tt.vis_spectrum(selection=selection)





def chebiRequest(chebiNb):
    o=requests.get(f'http://www.ebi.ac.uk/webservices/chebi/2.0/test/getCompleteEntity?chebiId={chebiNb}')
    tree = ET.ElementTree(ET.fromstring(o.text))
    r = tree.getroot()
    return {x.tag.split('}')[1]: x.text for x in list(r[0][0][0])}

x=np.arange(50, 1000)
cv=5 #ppm
for j in range(len(x)):
    out = []
    o1 = []
    for i in range(100):
        s = ((cv * 1e-6) * x[j])
        da = np.random.normal(x[j], s, 10)
        out.append(np.std(da))
        o1 = (np.quantile(da, 0.9) - np.quantile(da, 0.1))
    # plt.scatter(x[j], np.mean(o1), c='grey')
    plt.scatter(x[j], np.mean(o1), c='red')
    #plt.scatter(np.ones(100)*x[j], out)


import molmass

f = molmass.Formula('C4H8N2O3')
f.atoms
f.composition()
f.isotope
s=f.spectrum()

k=list(s.keys())
for i in range(len(k)):
    plt.scatter(s[k[i]][0], s[k[i]][1])




for i in range(nSpectra):
    mets = {x: res[i][x] for x in res[i] if not ('Id' in x)}
    q = con.execute(f"select * from PerSpectrumVariables WHERE Spectrum = {i}")
    specVar = [dict(x) for x in q.fetchall()]
    mets.update({vdict[k['Variable']]['PermanentName']: k['Value'] for k in specVar})
    edat[cvar[0]].append(mets)


qst ='SELECT * FROM Spectra as s JOIN AcquisitionKeys as ak ON s.AcquisitionKey=ak.Id WHERE Segment=3 AND ScanMode=5 ORDER BY Rt'
qst ='SELECT scanmode FROM Spectra as s JOIN AcquisitionKeys as ak ON s.AcquisitionKey=ak.Id WHERE Segment=3 ORDER BY Rt'

res=[dict(x) for x in con.execute(qst).fetchall()]

