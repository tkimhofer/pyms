
import sqlite3
import numpy as np
import pandas as pd

class ChemLibrary:

    def __init__(self, db_file):
        self.db_file = db_file
        self.con = sqlite3.connect(self.db_file)
        self.con.row_factory = sqlite3.Row

    def close(self):
        if self.con is not None:
            self.con.close()
            self.con = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def search_mim(self, mass=180.17, ppm=50.0):
        mass = float(mass)
        ppm = float(ppm)

        delta = ppm * 1e-6 * mass
        lo, hi = mass - delta, mass + delta

        query = """
            SELECT
                cd.compound_id,
                cp.ascii_name,
                cd.formula,
                cd.charge,
                cd.mass,
                cd.monoisotopic_mass,
                cp.stars
            FROM chemical_data cd
            LEFT JOIN compounds cp
                ON cd.compound_id = cp.id
            WHERE cd.monoisotopic_mass BETWEEN ? AND ?
        """

        res = pd.read_sql_query(query, self.con, params=(lo, hi))

        if not res.empty:
            res["diff_ppm"] = (
                np.abs(res["monoisotopic_mass"] - mass) / mass * 1e6
            )
            res["diff_ppm"] = res["diff_ppm"].round(3)

            res = res.sort_values(
                ["diff_ppm", "stars", "ascii_name"],
                ascending=[True, False, True],
            )

        print(f"MIM search {lo:.6f}–{hi:.6f}, {res.shape[0]} hits")
        return res

# # example scr:
# dbf = '/Users/tk/Downloads/chebi_embl.sql'
# db=ChemLibrary(dbf)
# d2 =db.search_mim(mass=180.064, ppm=100)




### this is how the example compund lib was established:
#  Here: EMBL-EBI CheBI - manually curated database of chemical entities
#  see: https://www.ebi.ac.uk/chebi/downloads

# def tab_to_sqlite(flatFile='/Users/tk/Downloads/chemical_data.tsv', sqlFile='/Users/tk/Downloads/chebi_embl.sql'):
#
#     import pandas as pd
#     import sqlite3
#     from pathlib import Path
#
#     df = pd.read_table(flatFile)
#     tbl_name = Path(flatFile).stem
#
#     with sqlite3.connect(sqlFile) as con:
#         df.to_sql(tbl_name, con, )
#
#     print(f'Added table "{tbl_name}" with {df.shape[0]:,} rows.')
#
# db_file = '.../chebi_embl.sqlite'
# tab_to_sqlite(flatFile='.../Downloads/chemical_data.tsv', sqlFile=db_file)
# tab_to_sqlite(flatFile='.../Downloads/compounds.tsv', sqlFile=db_file)


