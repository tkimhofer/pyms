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
        ds=pd.DataFrame(self.edat['Id'])
        self.scantime = ds.Rt

        polmap = {0: '(+)', 1: '(-)'}
        df1 = pd.DataFrame(
            [(polmap[x['Polarity']], str(x['MzAcqRangeLower']) + '-' + str(x['MzAcqRangeUpper']), x['MsLevel']) for x in
             self.edat['Id']], columns=['Polarity', 'MzAcqRange', 'MsLevel'])
        df1 = pd.DataFrame([self.csummary(i, df1) for i in range(df1.shape[1])], index=df1.columns).transpose()
        df1['nSc'] = len(self.edat['Id'])

        df2 = pd.DataFrame([(x['AcquisitionMode'], x['ScanMode'], x['Segment']) for x in self.edat['Id']],
                           columns=['AcqMode', 'ScMode', 'Segments'])
        t = pd.DataFrame(df2.groupby(['Segments']).apply(lambda x: self.vc(x, n=df2.shape[0])))
        t = t.rename(index={t.index[i]: 'Segment ' + str(t.index[i]) for i in range(t.shape[0])}).transpose()
        asps = 1 / ((self.scantime.max() - self.scantime.min()) / df1['nSc'].iloc[0])  # hz
        ii = np.diff(np.unique(self.scantime))
        sps = pd.DataFrame(
            {'ScansPerSecAver': asps, 'ScansPerSecMin': 1 / ii.max(), 'ScansPerSecMax': 1 / ii.min()}, index=[0])
        self.summary = pd.concat([df1, t, sps], axis=1).T
        print(self.summary)

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
        ct = x.value_counts(subset=['ScMode', 'AcqMode'])
        s = []
        for i in range(ct.shape[0]):
            s.append(
                f'{ct.index.names[0]}={ct.index[i][0]} & {ct.index.names[1]}={ct.index[i][1]}: {ct[ct.index[i]]} ({np.round(ct[ct.index[0]] / n * 100, 1)} %)')
        return '; '.join(s)


fname='/Users/TKimhofer/Desktop/RPN_Pl/other/covvac_C3_PLA_MS-AA_PAI03_VAXp41_280322_QC01_2.d'
# test=ExpSum(fname)
self=MSexp(fname)
self.plt_chromatogram(ctype=['bpc', 'tic'])
self.massSpectrumDIA(sid=3000, scantype='0_2_2_5_20.0')
self.vis_spectrum1(q_noise=0.95, selection={'mz_min': 0, 'mz_max': 1000, 'rt_min': 380, 'rt_max':395}, qcm_local=True)

self=MSexp(fpath)

pd.DataFrame([(x['AcquisitionMode'], x['ScanMode'], x['Segment']) for x in self.edat['Id']],
                           columns=['AcqMode', 'ScMode', 'Segments'])