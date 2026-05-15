from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import Union


@dataclass
class IsotopePeak:
    feature_id: Union[int, str]
    mz: float
    rt: float
    intensity: float
    isotope_index: int

@dataclass
class IsotopePattern:
    seed_id: Union[int, str]
    peaks: list[IsotopePeak]

@dataclass
class IsoPattern:
    mz: np.ndarray
    prob: np.ndarray

    @property
    def intensity(self):
        return self.prob / self.prob.max()

@dataclass
class IsotopeGroupingResult:
    table: pd.DataFrame
    patterns: dict[Union[int, str], IsotopePattern]

class IsotopePattern:
    def __init__(self, m_fid):
        self.c = 0
        self.fid = {f'{self.c}': m_fid}

    def add(self, m_fid):
        self.c += 1
        self.fid[f'{self.c}'] = m_fid
