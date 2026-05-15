
import numpy as np
import xml.etree.cElementTree as ET
import re

from msmate.io.helpers_xml import _children, _get_obo, _node_attr_recurse, _collect_spectra_chrom

def _read_mzml(self):
    """Extracts MS data from mzml file"""
    # this is for mzml version 1.1.0
    # schema specification: https://raw.githubusercontent.com/HUPO-PSI/mzML/master/schema/schema_1.1/mzML1.1.0.xsd
    # read in data, files index 0 (see below)
    # flag is 1/2 for msLevel 1 or 2
    tree = ET.parse(self.fpath)
    # tree = ET.parse(path)

    root = tree.getroot()
    child = _children(root)
    imzml = child.index('mzML')
    mzml_children = _children(root[imzml])
    obos = root[imzml][mzml_children.index('cvList')]  # controlled vocab (CV)
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
    self.out31 = _collect_spectra_chrom(
        s=run, d=20, c=0, flag=self.mslevel, tag='', obos=self.obo_ids,
        rt_min=self.scan_window.st_min,
        rt_max=self.scan_window.st_max
    )

    # self.out31 = _collect_spectra_chrom(
    #     s=run, d=20, c=0, flag="1", tag='', obos=self.obo_ids,
    #     rt_min=0,
    #     rt_max=10
    # )



# class ReadMT:
#     """mzML experiment class and import methods  (this is for targeted methods)"""
#     def __init__(self, fpath: str, mslev:str='targeted'):
#         """Import of XC-MS level 1 scan data from mzML files V1.1.0
#
#             Note that this class is designed for use with `MsExp`.
#
#             Args:
#                 fpath: Path `string` variable pointing to mzML file
#                 mslev: `String` defining ms level read-in (fulls can)
#         """
#         self.fpath = fpath
#         if mslev != 'targeted':
#             raise ValueError('Check mslev argument - This function currently support ms level 1 import only.')
#         self.mslevel = mslev
#         self._read_mzml()
#         self._createSpectMat()
#         self._stime_conv()
#
#
#     def _read_mzml(self):
#         """Extracts MS data from mzml file"""
#         # this is for mzml version 1.1.0
#         # schema specification: https://raw.githubusercontent.com/HUPO-PSI/mzML/master/schema/schema_1.1/mzML1.1.0.xsd
#         # read in data, files index 0 (see below)
#         # flag is 1/2 for msLevel 1 or 2
#         tree = ET.parse(self.fpath)
#         root = tree.getroot()
#         child = _children(root)
#         imzml = child.index('mzML')
#         mzml_children = _children(root[imzml])
#         obos = root[imzml][mzml_children.index('cvList')] # controlled vocab (CV
#         self.obo_ids = _get_obo(obos, obo_ids={})
#         seq = np.where(~np.isin(mzml_children, ['cvList', 'run']))[0]
#         pp = {}
#         for j in seq:
#             filed = _node_attr_recurse(s=root[imzml][j], d=4, c=0, ii=[])
#             dn = {}
#             for i in range(len(filed)):
#                 dn.update(dict(zip([filed[i]['path'] + '_' + re.sub('\{.*\}', '', x) for x in list(filed[i].keys())[1:]],
#                                    list(filed[i].values())[1:])))
#             pp.update({mzml_children[j]: dn})
#         run = root[imzml][mzml_children.index('run')]
#         self.out31 = _collect_spectra_chrom(s=run, ii={}, d=20, c=0, flag=self.mslevel, tag='', obos=self.obo_ids)
#
#
#
#     def prep_df(self, df):
#
#         if 'MS:1000127' in df.columns:
#             add = np.repeat('centroided', df.shape[0])
#             add[~(df['MS:1000127'] == True)] = 'profile'
#             df['MS:1000127'] = add
#
#         if 'MS:1000128' in df.columns:
#             add = np.repeat('profile', df.shape[0])
#             add[~(df['MS:1000128'] == True)] = 'centroided'
#             df['MS:1000128'] = add
#
#         if "time_unit" not in df.columns:
#             df["time_unit"] = "sec"
#
#         df = df.rename(
#             columns={'UO:0000010': 'time_unit', 'UO:0000031': 'time_unit', 'defaultArrayLength': 'n',
#                      'MS:1000129': 'polNeg', 'MS:1000130': 'polPos',
#                      'MS:1000127': 'specRepresentation', 'MS:1000128': 'specRepresentation',
#                      'MS:1000505': 'MaxIntensity', 'MS:1000285': 'SumIntensity',
#                      'MS:1000016': 'Rt'
#                      })
#         df.columns = [self.obo_ids[x]['name'].replace(' ', '_') if x in self.obo_ids.keys() else x for x in
#                            df.columns.values]
#         df['fname'] = self.fpath
#         return df
#
#     def _createSpectMat(self):
#         # tyr targeted data has two dimensions: 1. rt, 2. intensity
#         """Organise raw MS data and scan metadata"""
#         sc_msl1 = [(i, len(x['data']['Int']['d'])) for i, x in self.out31.items() if len(x['data']) == 2] # sid of ms level 1 scans
#
#         self.df = pd.DataFrame([x['meta'] for i, x in self.out31.items() if len(x['data']) == 2])
#         self.df['defaultArrayLength'] = self.df['defaultArrayLength'].astype(int)
#
#         from collections import defaultdict, Counter
#         xrawd = {}
#         if 'MS:1000129' in self.df.columns:
#             nd = self.df['defaultArrayLength'][self.df['MS:1000129'] == True].sum()
#             xrawd['1N'] = np.zeros((4, nd))
#
#         if 'MS:1000130' in self.df.columns:
#             nd = self.df['defaultArrayLength'][self.df['MS:1000130'] == True].sum()
#             xrawd['1P'] = np.zeros((4, nd))
#
#         row_counter = Counter({'1P': 0, '1N': 0})
#         sid_counter = Counter({'1P': 0, '1N': 0})
#         dfd = defaultdict(list)
#         for i, s in enumerate(sc_msl1):
#
#             d = self.out31[s[0]]
#             if s[1] != self.df['defaultArrayLength'].iloc[i]:
#                 raise ValueError('Check data extraction')
#
#             cbn = [[d['meta']['index']] * s[1]]
#             for k in d['data']:
#                 cbn.append(d['data'][k]['d'])
#
#             if 'MS:1000129' in d['meta']:
#                 fstr = '1N'
#             elif 'MS:1000130' in d['meta']:
#                 fstr = '1P'
#
#             cbn.append([sid_counter[fstr]] * s[1])
#             add = np.array(cbn)
#             xrawd[fstr][:, row_counter[fstr]:(row_counter[fstr] + s[1])] = add
#
#             dfd[fstr].append(d['meta'])
#
#             sid_counter[fstr] += 1
#             row_counter[fstr] += (s[1])
#
#         for k, d, in dfd.items():
#             print(k)
#             dfd[k] = self.prep_df(pd.DataFrame(dfd[k]))
#
#         self.dfd = dfd
#         self.xrawd = xrawd
#         self.ms0string = row_counter.most_common(1)[0][0]
#         self.ms1string = None
#
#
#     def _stime_conv(self):
#         pass
#         """Performs scan time conversion from minutes to seconds."""
#         for k, d, in self.dfd.items():
#             if 'min' in d['time_unit'].iloc[1]:
#                 self.dfd[k]['Rt'] = d['Rt'].astype(float) * 60
#                 self.dfd[k]['time_unit'] = 'sec'
#                 self.xrawd[k][3] = self.xrawd[k][3] * 60
