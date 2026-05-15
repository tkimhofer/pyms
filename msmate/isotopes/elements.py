import numpy as np
import pandas as pd

class Element:
    def __init__(self, eid, iId, iM, iA):
        # element id, isotope IDs, isotopic masses, isotopic abundance (fraction)
        self.id = eid
        self.isotopes = iId
        self.iM = iM
        self.iA = iA

        if np.sum(self.iA) != 1:
            raise ValueError('probs do not sum up to 1')


class ElementTable:
    def __init__(self, dd_path='Atomic Weights and Isotopic Compositions.csv'):
        dd = pd.read_csv(dd_path, comment='#')
        dd = dd[~dd['Isotopic Composition'].isnull()]
        self.data = dd
        l = self.data.alias.unique()
        for i in range(len(l)):
            id = list(l)[i]
            sub = self.data[self.data.alias == l[i]]
            try:
                s = Element(id, sub['Mass Number'].values, sub['Relative Atomic Mass'].values,
                            sub['Isotopic Composition'].values)
                setattr(self, id, s)
            except:
                print(f'Skipping {id}')
                pass
