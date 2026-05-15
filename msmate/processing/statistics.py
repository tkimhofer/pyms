from typeguard import typechecked
import numpy as np
import matplotlib.pyplot as plt
from msmate.core.types import  IDX_INT

@typechecked
class MSstat:
    """ Base class defining subsetting & statistical functions operating on 2D MS data.

    Note that this class is designed for use with `MsExp`.
    """

    def ecdf(self, plot: bool = False):
        """Calculate empirical cumulative distribution function (ECDF)"""

        x = self.xrawd[self.ms0string][IDX_INT]
        x = np.sort(x)  # critical!

        y = np.arange(1, len(x) + 1) / len(x)

        self.ecdf_x = x
        self.ecdf_y = y

        if plot:
            # import matplotlib.pyplot as plt

            f, ax = plt.subplots(1, 1)

            ax.plot(x, y)
            ax.set_xscale("log")  # better readability

            ax.set_ylabel('P(X ≤ x)')
            ax.set_xlabel('Intensity')

            if hasattr(self, 'noise_thres'):
                ax.axvline(self.noise_thres, color='red', linestyle='--')

            plt.show()

    def get_ecdfInt(self, p):
        """Inverse ECDF (quantile function)"""

        p = np.asarray(p)

        if np.any((p < 0) | (p > 1)):
            raise ValueError("p must be in [0, 1]")

        x = self.ecdf_x
        y = self.ecdf_y

        idx = np.searchsorted(y, p, side="left")

        idx = np.clip(idx, 0, len(x) - 1)

        return x[idx]

