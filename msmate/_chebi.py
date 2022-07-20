import numpy as np
import pandas as pd
import re, sqlite3, requests

class chebic:
    # class stores single chemical from chebi
    # retrieves further information from chebi (e.g. chemical descriptors for retention time prediction)
    def __init__(self, dbFile, cid='24999', ):
        self.dbFile = dbFile
        self.c = sqlite3.connect(self.dbFile)
        self.c.row_factory = sqlite3.Row

        self.cid = cid

        self.getEntry()
        self.getParent()
        self.getSynonyms()
        self.get_ChemicalData()

        if self.parent is not None:
            self.cid = self.parent
            self.getEntry()
            self.getParent()
            self.getSynonyms()
            self.get_ChemicalData()
            self.getAcc()

    def __del__(self):
        print('c closed 1')
        self.c.close()

    def getParent(self):
        self.parent = self.entry['parent_id']
        print(self.parent)
        # self.parent = self.c.execute(f'select parent_id from compounds where id = {self.cid}').fetchall()
        # print(self.parent)

    def getEntry(self):
        ent = self.c.execute(f'select * from compounds where id = "{self.cid}"').fetchall()
        if len(ent) > 1:
            raise ValueError('got more than one entry for this id')
        self.entry = dict(ent[0])
        print(self.entry)

    def getSynonyms(self):
        self.synonyms = pd.DataFrame([dict(x) for x in self.c.execute(f'select * from names where compound_id = "{self.cid}"').fetchall()])
        print(self.synonyms)

    def get_ChemicalData(self):
        out = [dict(x) for x in self.c.execute(f'select * from chemical_data where compound_id = "{self.cid}"').fetchall()]
        self.chemData = pd.DataFrame(out).pivot(index='compound_id', columns='type', values='chemical_data')
        print(self.chemData)

    def getAcc(self):
        out = [dict(x) for x in
               self.c.execute(f'select * from database_accession where compound_id = "{self.cid}"').fetchall()]
        self.acc = pd.DataFrame(out)

    def getComm(self):
        out = [dict(x) for x in
               self.c.execute(f'select * from comments where compound_id = "{self.cid}"').fetchall()]
        self.comm = pd.DataFrame(out)

    def getOrig(self):
        out = [dict(x) for x in
               self.c.execute(f'select * from compound_origins where compound_id = "{self.cid}"').fetchall()]
        self.orig = pd.DataFrame(out)

    def getChebiOnline(self):
        url = 'https://www.ebi.ac.uk/ols/api/search?ontology=chebi' + \
              '&exact=True' + '&q=' + self.entry['name'] + \
              '&rows=' + str(int(10))

        response = requests.get(url)
        if response.status_code == 200:
            self.chebiRresp = response.json()

# def getChemDesc(self):
    #     q='https://www.ebi.ac.uk/chembl/api/data/molecule.json?molecule_properties__mw_monoisotopic__range=180,181'
    #     response = requests.get(q)
    #     response.raise_for_status()
    #     print(response.status_code)
    #     tt=xmltodict.parse(response.text)
    #     tt.keys()
    #     tt.response
    #     data = response.json()

class chemDb:
    @staticmethod
    def regexp(expr, item):
        reg = re.compile(expr, re.IGNORECASE)
        return reg.search(item) is not None

    def __init__(self, dbFile):
        self.dbFile = dbFile
        self.c = sqlite3.connect(self.dbFile)
        self.c.row_factory = sqlite3.Row
        self.c.create_function("REGEXP", 2, self.regexp)

    def __del__(self):
        print('c closed')
        self.c.close()

    def searchName(self, name='Hippurate', lang='en'):
        res = pd.read_sql_query(f'select * from compounds where name regexp "{name}"', self.c)
        res1 = pd.read_sql_query(f'select * from names where name regexp "{name}" and language = "{lang}"', self.c)
        # check for exact match
        out=pd.DataFrame({'cid': res.id.values.tolist() + res1.compound_id.values.tolist(), \
                      'name': res.name.values.tolist() + res1.name.values.tolist(), \
                      'source': res.source.tolist() + res1.source.tolist(), \
                      'type': np.repeat('preferred name', res.shape[0]).tolist() + res1.type.tolist()
                      }).sort_values('name')
        return out

    def searchMim(self, mass=180, ppm=50):
        d = (ppm * 1e-6 * mass) / 2
        ra = (mass - d, mass + d)
        res = pd.read_sql_query(f'select cd.compound_id, cpd.name, f.formula, cd.type, cd.chemical_data, cpd.source, cpd.status, '
                                f'cpd.definition, cpd.created_by, cpd.modified_on, cpd.star '
                                f'from chemical_data as cd '
                                f'JOIN compounds as cpd ON cpd.id=cd.compound_id '
                                f'LEFT JOIN formula as f ON f.compound_id=cd.compound_id '
                                f'where cd.type="MONOISOTOPIC MASS" and cd.chemical_data>={ra[0]} and cd.chemical_data<={ra[1]}', self.c)
        res['diff_ppm'] = np.round((res.chemical_data-180)/180 * 1e6)
        res=res.sort_values(['diff_ppm', 'star', 'name'], ascending=[True, False, True])
        print(f'MIM search {ra}, {res.shape[0]} db hits')
        return res


# # example scr:
# dbf = '/Users/TKimhofer/pyt/pyms/chebi1.db'
# db=chemDb(dbf)
# d2 =db.searchMim(mass=180.06)
# d1 =db.searchName(name='hipp')

# id = '606564'
# ch=chebic(dbf, cid=id, )
# ch1=chebic(cid=ch.compound_id.iloc[0])



#######
# sqlite db established w the following code
# ### establish sqlite db for compound data from chebi
# def sqlToSqlite(dbPath='/Users/TKimhofer/Downloads/generic_dump_allstar/compounds.sql', sqlPath='/Users/TKimhofer/pyt/pyms/chebi6.db'):
#     import os; import sqlite3; import re
#     tbl = os.path.basename(dbPath).split('.')[0]
#
#     qs = open(dbPath, 'r').read()
#     qss = qs.split('\ninsert')
#
#     st = qss[1].split(' into ')[1].split(' (')
#     tblName = f'{st[0]}'
#     tblVars = f'({st[1].split(")")[0]})'
#     tblVars = re.sub('\Wid\W', '(id int not null primary key,', tblVars) # adds primary key constraints
#
#
#     if tblName in 'compounds':
#         tblVars = re.sub('\)$', ', foreign key (parent_id) references compounds (id))', tblVars)
#
#     if tblName in 'comments database_accession names structures reference compound_origins ':
#         tblVars = re.sub('\)$', ', foreign key (compound_id) references compounds (id))', tblVars)
#
#     if tblName in 'relation':
#         tblVars = re.sub('\)$', ', foreign key (init_id) references compounds (id),  foreign key (final_id) references compounds (id))', tblVars)
#
#
#     c = sqlite3.connect(sqlPath)
#     # c.execute('select * from sqlite_master where type="table"').fetchall()
#     # tt=c.execute('select count(*) from compounds').fetchall()
#
#     tblVars
#     try:
#         with c:
#             c.execute(f'create table {tblName}{tblVars};')
#             pfx = lambda x: f'insert{x}'
#             [c.execute(pfx(d)) for d in qss if d != '']
#             c.commit()
#
#             a = c.execute(f'select count(*) from {tblName}')
#             print(f'Added table "{tblName}" and filled with {a.fetchall()[0][0]} rows')
#             print('---')
#         c.close()
#     except Exception as e:
#         print(e)
#         c.close()
#
# sqlToSqlite(dbPath='/Users/TKimhofer/Downloads/generic_dump_allstar/compounds.sql', sqlPath='/Users/TKimhofer/pyt/pyms/chebi1.db') # main table
# sqlToSqlite(dbPath='/Users/TKimhofer/Downloads/generic_dump_allstar/chemical_data.sql', sqlPath='/Users/TKimhofer/pyt/pyms/chebi1.db') # foreign key to compounds
# sqlToSqlite(dbPath='/Users/TKimhofer/Downloads/generic_dump_allstar/comments.sql', sqlPath='/Users/TKimhofer/pyt/pyms/chebi1.db') # foreign key to compounds
# sqlToSqlite(dbPath='/Users/TKimhofer/Downloads/generic_dump_allstar/database_accession.sql', sqlPath='/Users/TKimhofer/pyt/pyms/chebi1.db') # foreign key to compounds
# sqlToSqlite(dbPath='/Users/TKimhofer/Downloads/generic_dump_allstar/names.sql', sqlPath='/Users/TKimhofer/pyt/pyms/chebi1.db') # foreign key to compounds
# sqlToSqlite(dbPath='/Users/TKimhofer/Downloads/generic_dump_allstar/compound_origins.sql', sqlPath='/Users/TKimhofer/pyt/pyms/chebi1.db') # foreign key to compounds
# sqlToSqlite(dbPath='/Users/TKimhofer/Downloads/generic_dump_allstar/relation.sql', sqlPath='/Users/TKimhofer/pyt/pyms/chebi1.db') # foreign key to compounds
# # TODO: omitting large tables: references, structures  # foreign key to compounds
# # sqlToSqlite(dbPath='/Users/TKimhofer/Downloads/generic_dump_allstar/structures.sql', sqlPath='/Users/TKimhofer/pyt/pyms/chebi1.db') # foreign key to compounds
# # sqlToSqlite(dbPath='/Users/TKimhofer/Downloads/generic_dump_allstar/references.sql', sqlPath='/Users/TKimhofer/pyt/pyms/chebi1.db') # foreign key to compounds
#
#
# # TODO: separating formula from other chemical data that is numeric
# sqlPath='/Users/TKimhofer/pyt/pyms/chebi1.db'
# c = sqlite3.connect(sqlPath)
#
# df = pd.read_sql_query('select * from chemical_data', c)
# idc = df.type == 'FORMULA'
#
# dff = df[idc].copy()
# ddf1=dff.rename(columns={'chemical_data': 'formula'})
#
# dfn = df[~idc].copy()
# dfn.chemical_data = dfn.chemical_data.astype(float)
#
# c.execute('drop table chemical_data')
# c.execute('drop table formula')
# c.commit()
#
# dfn.to_sql('chemical_data', c, index=False)
# ddf1.to_sql('formula', c, index=False)
# c.commit()
# c.close()