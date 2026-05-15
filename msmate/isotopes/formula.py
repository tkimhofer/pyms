### placeholder for formula parsing
import re

def split_formula(formula:str):
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    return [(e, int(n) if n else 1) for e, n in tokens]



# formula = 'C3H6O3'
# formula = 'C6H12O6'
# formula = 'H2O'
# formula = 'CH4'
# formula = 'NaCl'
# formula = 'C'
# formula = 'C10H16N5O13P3'