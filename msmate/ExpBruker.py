# get summary of Bruker experiments w sqlite
import sqlite3

import pandas as pd


# self = ExpInfo(x)

class ExpInfo:
    def __init__(self, dpath, dbfile, c):
        import sqlite3
        self.expfolder = dpath
        self.dbfile = dbfile
        self.c = c
        self.getScanInfo()
        self.getExpInfo()
    @staticmethod
    def csummary(i, df):
        import numpy as np
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
        ct = x.value_counts(subset=['ScMode', 'AcqMode'])
        s = []
        for i in range(ct.shape[0]):
            s.append(
                f'{ct.index.names[0]}={ct.index[i][0]} & {ct.index.names[1]}={ct.index[i][1]}: {ct[ct.index[i]]} ({np.round(ct[ct.index[0]] / n * 100, 1)} %)')
        return '; '.join(s)
    def getScanInfo(self):
        import sqlite3
        q = "SELECT * FROM Spectra as s JOIN AcquisitionKeys as ak ON s.AcquisitionKey=ak.Id"
        self.c.row_factory = sqlite3.Row
        res = [dict(x) for x in self.c.execute(q).fetchall()]
        df1 = pd.DataFrame([(x['Polarity'],  str(x['MzAcqRangeLower'])+'-'+str(x['MzAcqRangeUpper']), x['MsLevel']) for x in res], columns=['Polarity', 'MzAcqRange', 'MsLevel'])
        df1=pd.DataFrame([self.csummary(i, df1) for i in range(df1.shape[1])], index=df1.columns).transpose()
        df1['nSc'] = len(res)
        df2 =pd.DataFrame([(x['AcquisitionMode'], x['ScanMode'], x['Segment']) for x in res],
                     columns=['AcqMode', 'ScMode', 'Segments'])
        t=pd.DataFrame(df2.groupby(['Segments']).apply(lambda x: self.vc(x, n=df2.shape[0])))
        t=t.rename(index= {i: 'Segment '+str(i) for i in range(1, 1+t.shape[0])}).transpose()
        self.scanSummary = pd.concat([df1, t], axis=1)
    def getExpInfo(self):
        import numpy as np
        props = pd.DataFrame([dict(x) for x in self.c.execute('select * from Properties').fetchall()]).T
        props=props.rename(columns={i: props.loc['Key'].values[i] for i in range(props.shape[1])}).loc['Value']
        dd=pd.DataFrame([dict(x) for x in self.c.execute(f"select * from PerSpectrumVariables").fetchall()])
        ddg=dd.groupby('Variable')
        to=[self.csummary(2, dd.iloc[ddg.groups[list(ddg.groups.keys())[i]]]) for i in range(len(ddg.groups.keys()))]
        q = self.c.execute(f"select * from SupportedVariables")
        vdict = {x[0]: x for x in q.fetchall()}
        perspeV=pd.DataFrame(to, index=[vdict[i][1] for i in np.unique(dd.Variable)])
        self.fullExpSummary = pd.concat([self.scanSummary.T, perspeV, props], axis=0)

class TechRun:
    def __init__(self, dpath):
        import os
        import subprocess
        import shlex
        self.dpath = dpath
        if not os.path.isdir(self.dpath):
            raise ValueError('Db file not found - check path')
        cmd = f'find {shlex.quote(self.dpath)} -depth 1 -type d -regex \'.*d$\' -print'
        sp = subprocess.getoutput(cmd)
        self.dpathExp = [x for x in sp.split('\n') if 'Sync' not in x]
        self._getExpInfo()
    def _getExpInfo(self):
        import numpy as np
        import os
        import sqlite3
        self.oo=[]
        for x in self.dpathExp:
            if os.path.isdir(x):
                fb= os.path.join(x, "analysis.sqlite")
                print(fb)
                if os.path.isfile(fb):
                    try:
                        print(fb)
                        c = sqlite3.connect(fb)
                        tbl=c.execute('select name from sqlite_master where type="table" and name in ("Spectra", "Properties", "PerSpectrumVariables", "SupportedVariables")').fetchall()
                        print(tbl)
                        if len(tbl) == 4:
                            print(x)
                            self.oo.append(ExpInfo(x, fb, c).fullExpSummary)
                            print('--')
                    except:
                        print(f'Can\'t open: {fb}')
                        continue
                else:
                    print(f'Skipping {x}, no db file found')
        self.summaryTbl = pd.concat(self.oo, axis=1).T
        self.summaryTbl.AcquisitionDateTime = pd.to_datetime(self.summaryTbl.AcquisitionDateTime, errors='coerce')
        self.summaryTbl=self.summaryTbl.sort_values('AcquisitionDateTime')
        self.summaryTbl = self.summaryTbl.iloc[np.argsort(self.summaryTbl.AcquisitionDateTime.rank().astype(int))]
        self.summaryTbl = self.summaryTbl.reset_index(drop=True)

        import matplotlib.pyplot as plt
        plt.plot(self.summaryTbl.AcquisitionDateTime, [float(x.split(' ')[0]) for x in self.summaryTbl.TOF_DeviceTempCurrentValue1])
        plt.plot(self.summaryTbl.AcquisitionDateTime, [float(x.split(' ')[0]) for x in self.summaryTbl.TOF_DeviceTempCurrentValue2])



        #self.summaryTbl = self.summaryTbl.set_axis(self.dpathExp, axis=0)




dpath='/Users/TKimhofer/Desktop/RPN_Pl/'
self = TechRun(dpath)

dpath = '/Users/TKimhofer/Desktop/AA BARWON'
a=TechRun(dpath)

dpath = '/Volumes/alims4ms/PAI05/barwin20/Amino Acid'


aa=a.summaryTbl


aa=self.summaryTbl
