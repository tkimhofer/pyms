from typeguard import typechecked
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

from msmate.core.types import ScanWindow
from msmate.core.helpers import scan_window_to_dict
from msmate.io.mzml import _read_mzml
from msmate.io.mzxml import _read_mzxml

@typechecked
class ReadM:
    """mzML experiment class and import methods"""
    def __init__(self, fpath: str, scan_window: ScanWindow, mslev:str='1'):
        """Import of XC-MS level 1 scan data from mzML files V1.1.0

            Note that this class is designed for use with `MsExp`.

            Args:
                fpath: Path `string` variable pointing to mzML file
                mslev: `String` defining ms level read-in (fulls can)
        """
        self.fpath = fpath
        self.scan_window = scan_window
        self.mslevel = mslev

        if mslev != '1':
            raise ValueError('Check mslev argument - This function currently support ms level 1 import only.')

        self.import_params = {
            "mslevel": self.mslevel,
            "scan_window": scan_window_to_dict(self.scan_window),
            "time_unit_internal": "sec",
        }

        suffix = Path(self.fpath).suffix.lower()

        if suffix == ".mzml":
            self.read_mzml()
        elif suffix == ".mzxml":
            self.read_mzxml()
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        self._createSpectMat()

    def prep_df(self, df):

        if 'MS:1000127' in df.columns:
            add = np.repeat('centroided', df.shape[0])
            add[~(df['MS:1000127'] == True)] = 'profile'
            df['MS:1000127'] = add

        if 'MS:1000128' in df.columns:
            add = np.repeat('profile', df.shape[0])
            add[~(df['MS:1000128'] == True)] = 'centroided'
            df['MS:1000128'] = add

        if "time_unit_ori" not in df.columns:
            df["time_unit_ori"] = "unknown"

        df = df.rename(
            columns={
                     'defaultArrayLength': 'n',
                     'MS:1000129': 'polNeg',
                     'MS:1000130': 'polPos',
                     # 'MS:1000127': 'specRepresentation',
                     # 'MS:1000128': 'specRepresentation',
                     'MS:1000505': 'MaxIntensity',
                     'MS:1000285': 'SumIntensity',
                     'MS:1000016': 'Rt'
                     })

        rep_cols = [c for c in ['MS:1000127', 'MS:1000128'] if c in df.columns]
        if rep_cols:
            df['specRepresentation'] = df[rep_cols[0]]
            df = df.drop(columns=rep_cols)

        if hasattr(self, "obo_ids") and self.obo_ids:
            df.columns = [
                self.obo_ids[x]['name'].replace(' ', '_') if x in self.obo_ids else x
                for x in df.columns
            ]

        df['fname'] = self.fpath

        return df

    def _createSpectMat(self):
        """Organise raw MS data and scan metadata"""

        spectra = [(sid, x)
                   for sid, x in self.out31.items()
                   if (len(x['data']) == 2) and ('scan' in x['meta'].get('id', ''))
                   ]
        if len(spectra) == 0:
            raise ValueError("No MS1 spectra found after filtering.")

        n_points = Counter({'1P': 0, '1N': 0})
        for _, x in spectra:
            meta = x["meta"]
            inten = x["data"]["Int"]["d"]

            if "MS:1000129" in meta:
                fstr = "1N"
            elif "MS:1000130" in meta:
                fstr = "1P"
            else:
                fstr = "1P"
                meta["polarity_inferred"] = True

            n_points[fstr] += len(inten)

        xrawd = {} # esi polarity in psi-ms cv
        for fstr, n in n_points.items():
            if n > 0:
                xrawd[fstr] = np.zeros((5, n), dtype=float)

        row_counter = Counter({'1P': 0, '1N': 0})
        sid_counter = Counter({'1P': 0, '1N': 0})
        dfd = defaultdict(list)

        for sid, x in spectra:
            meta = x["meta"]
            mz = x["data"]["m/z"]["d"]
            inten = x["data"]["Int"]["d"]

            iLen = len(mz)

            if iLen != len(inten):
                raise ValueError(f"m/z and intensity length mismatch in scan {sid}")

            # after mz filtering, defaultArrayLength should equal filtered length
            meta["defaultArrayLength"] = iLen

            if "MS:1000129" in meta:
                fstr = "1N"
            elif "MS:1000130" in meta:
                fstr = "1P"
            else:
                fstr = "1P"
                meta["polarity_inferred"] = True

            dfd[fstr].append(meta)

            start = row_counter[fstr]
            stop = start + iLen

            xrawd[fstr][:, start:stop] = np.vstack([
                np.full(iLen, meta["index"], dtype=float),  # scanIdOri
                mz.astype(float),  # mz
                inten.astype(float),  # intensity
                np.full(iLen, meta["MS:1000016"], dtype=float),  # scantime seconds
                np.full(iLen, sid_counter[fstr], dtype=float),  # scanIdNorm
            ])

            sid_counter[fstr] += 1
            row_counter[fstr] = stop

        for k in list(dfd.keys()):
            dfd[k] = self.prep_df(pd.DataFrame(dfd[k]))

        self.dfd = dfd
        self.xrawd = xrawd

        # choose polarity with most points
        self.ms0string = row_counter.most_common(1)[0][0]
        self.ms1string = None


    def read_mzml(self):
        _read_mzml(self)
    def read_mzxml(self):
        _read_mzxml(self)