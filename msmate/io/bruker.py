import pandas as pd
import numpy as np
import os
import docker as dock
import time
import pickle

class ReadBruker:
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

