import os.path

import astropy.units as u
import numpy as np
import pandas
from astropy.table import Table

from soma.generators import Generator


class DC2Generator(Generator):
    """
    Generate samples from the DC2 catalog. Only floating point fields are considered, and prior
    pruning of columns and rows is done (i.e., remove columns with all NaN or rows with any NaN)

    Parameters
    ----------
    dataset : str
        Path to the FITS catalog
    snr : [Optional] float
        If present, apply a Signal-to-Noise filtering over the VIS band (FLUX / ERROR >= snr)
    abs_mag : [Optional] float
        If present, apply a filtering over the brightness on the VIS band

    See Also
    --------
    G. Euclid Collaboration: Desprez et al., “Euclid preparation. X. The euclid photometric-redshift challenge,”
    Astronomy & Astrophysics, vol. 644, p. A31, Dec. 2020, doi: 10.1051/0004-6361/202039403.
    """

    @staticmethod
    def __filter_snr(df: pandas.DataFrame, snr: float):
        mask = df['FLUX_VIS'] / df['FLUXERR_VIS'] < snr
        df.drop(df[mask].index, inplace=True)

    @staticmethod
    def __filter_abs_mag(df: pandas.DataFrame, abs_mag: float):
        mask = df['FLUX_VIS'] < (abs_mag * u.ABmag).to(u.uJy).value
        df.drop(df[mask].index, inplace=True)

    @staticmethod
    def __filter_constant(df: pandas.DataFrame):
        mask = (df == df.iloc[0]).all()
        df.drop(df.columns[mask], axis=1, inplace=True)

    def __init__(self, dataset: str = os.path.expanduser('~/Work/Data/DC2/euclid_cosmos_DC2_S2_v2.1_calib.fits'), *,
                 snr: float = None, abs_mag: float = None):
        df = Table.read(dataset).to_pandas()
        df.replace(-99, np.nan, inplace=True)
        df.replace(-99.9, np.nan, inplace=True)
        with pandas.option_context('mode.use_inf_as_na', True):
            df.dropna(axis=1, how='all', inplace=True)
            df.dropna(axis=0, how='any', inplace=True)
        if snr:
            self.__filter_snr(df, snr)
        if abs_mag:
            self.__filter_abs_mag(df, abs_mag)
        self.__filter_constant(df)
        self.__data = df.select_dtypes(include=[np.float32, np.float64]).to_numpy()

    @property
    def dimensions(self) -> int:
        return self.__data.shape[1]

    @property
    def array(self) -> np.ndarray:
        return self.__data

    def sample(self, n: int) -> np.ndarray:
        idxs = np.random.choice(len(self.__data), n)
        return self.__data[idxs]
