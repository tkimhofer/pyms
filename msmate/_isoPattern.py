# calculate isotopic distribution of a molecular formula
# define classes for each element
class element:
    def __init__(self, eid, iId, iM, iA):
        # element id, isotope IDs, isotopic masses, isotopic abundance (fraction)
        import numpy as np
        self.id = eid
        self.isotopes = iId
        self.iM = iM
        self.iA = iA

        if np.sum(self.iA) != 1:
            raise ValueError('probs do not sum up to 1')

class element_list:
    def __init__(self, dd_path='isoDist.xlsx'):
        import pandas as pd
        import pandas as pd
        dd = pd.read_excel(dd_path)

        self.data = dd
        l = self.data.Element.unique()
        for i in range(len(l)):
            id = list(l)[i]
            sub = self.data[self.data.Element == l[i]]
            try:
                s=element(id, sub['Neutrons'].values, sub['massRelAtomic'].values, sub['isotopicComp'].values)
                setattr(self, id, s)
            except:
                pass
# el=element_list()

class calc_mzIsotopes:
    def __init__(self, formula, el):
        self.formula = formula
        self.el = el
        self.prep_formula()
        self.elementary_masses()
        self.molecule_Iprediction()
        
    def prep_formula(self):
        import re
        res = re.split('(\d+)', self.formula)
        self.elem = res[::2][0:-1]
        self.stoi = [int(x) for x in res[1::2]]

    def elementary_masses(self):
        import numpy as np
        import itertools

        # average atomic mass: weighted average: weight of elementary isotope weighted by natural abundance
        e_comp = {}
        exact_mass = 0
        m_av = 0
        m_mlp = 1
        for i in range(len(self.elem)):
            
            eobj = getattr(self.el, self.elem[i] )
            n_I = len(eobj.isotopes)

            # how many permutations
            n_I ** self.stoi[i]
            
            # create index string for itertools fct
            cbn_comb = [''.join([str(i) for i in np.arange(n_I)])][0]
            perms = list(itertools.product(cbn_comb, repeat=int(self.stoi[i])))
            #e_perms.append(perms)

            # calculate total mass and likelihood of accurende
            m = [eobj.iM[list(map(int, i))].sum() for i in perms]
            l =  [eobj.iA[list(map(int, i))].prod() for i in perms]

            m_av += (np.sum(eobj.iM * eobj.iA) * self.stoi[i]) # average mass for all elements in molecule

            # calculate mass difference from mass with highest probability (=monoisotopic mass)
            im = np.argmax(l)
            ml = m[im]
            m_delta = m - ml

            e_comp.update({self.elem[i]: {'mlp': l[im], 'ml':ml, 'm_d': m_delta, 'm': m, 'l':l,  'cbn': [''.join(x) for x in perms], 'eobj': eobj}})

            exact_mass += ml
            m_mlp *= l[im]

        self.comp_data = e_comp
        self.mass_exact = exact_mass
        self.mass_average = m_av
        self.mass_exact_likellihood = np.prod(m_mlp)
        
        print(f'Nominal mass (M): {np.round(self.mass_exact)} u')
        print(f'Average mass: {np.round(self.mass_average, 4)} u')
        print(f'Exact/Monoisotopic mass: {np.round(self.mass_exact, 6)} u (~{np.round(self.mass_exact_likellihood, 4)}%)')

    def molecule_Iprediction(self):
        import itertools
        import numpy as np
        import pandas as pd
        
        s = [self.comp_data[x]['cbn'] for x in self.comp_data.keys()]
        comb = list(itertools.product(*s))
        
        md = np.zeros((4, len(comb)))
        for i in range(len(comb)): # for each combination
            x_s = comb[i]
            m = 0
            m_d = 0
            l = 1
            for j in range(len(x_s)): # get mass, mass difference to exact/monoitotopic mass and likelihood 
                id = self.elem[j]
                st = x_s[0]
                idx = self.comp_data[id]['cbn'].index(x_s[j])
                m_d += self.comp_data[id]['m_d'][idx]
                m += self.comp_data[id]['m'][idx]
                l *= self.comp_data[id]['l'][idx]

            md[0:4,i] = [m, l, m_d, int(np.round(m_d))]
            
        # get mass differences
        
        mplus = np.unique(md[3])
        prob_mplus = []
        average_mass = []
        cbn = []
        for i in range(len(mplus)):
            idx = np.where(md[3] == mplus[i])[0]
            prob_mplus.append(np.sum(md[1,idx]))
            average_mass.append(np.sum(md[1,idx]/np.sum(md[1,idx]) * md[0, idx]))
            cbn.append(len(idx))

        idc_keep = np.where(np.array(prob_mplus) > (1e-7))[0]
        out=pd.DataFrame({'molecular ion': ['[M]+' + str(int(x)) for x in mplus], 'average mass': [str(np.round(x, 4)) for x in average_mass], 'prob': np.round(prob_mplus, 7)})
        out['abundance (%)'] = out.prob / out.prob.max() * 100
        out['n_cbn'] = cbn

        out.loc[0, 'molecular ion']= '[M]'
        out.index = out['molecular ion']
        out.index

        self.isotopesAbundance = out[['average mass', 'abundance (%)', 'prob', 'n_cbn']]
        self.isotopesAbundance_reduced =  self.isotopesAbundance.iloc[idc_keep]
        
        print(f'\nEstimated isotopic distribution pattern for {self.formula}:\n')
        print(self.isotopesAbundance_reduced)
        
        le_om = self.isotopesAbundance.shape[0]-self.isotopesAbundance_reduced.shape[0]
        if le_om > 0:
            print(f'{le_om} entries omitted where p < {1e-7}')

# isotopes(formula, el)
# plt.scatter(mplus, average_mass)


# class adducts:
#     def __init__(self, isotopes, mode, ):
#         self.isotopes = isotopes




# find mz isotopes