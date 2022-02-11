import os.path
from typing import Tuple

import numpy as np

from soma.generators import Generator

_age_dataset_dir = os.path.join(os.path.dirname(__file__), 'data', 'age')


class AgeGenerator(Generator):
    """
    Generate samples from the projections into the last hidden layer from the neural
    network published in

    DEX: Deep EXpectation of apparent age from a single image

    URL: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/

    Note that the projections are generated offline to avoid repeating the work

    Parameters
    ----------
    min_age : int
        Minimum age (included)
    max_age : int
        Maximum age (excluded)
    """

    DATASET: np.ndarray = None

    @classmethod
    def load_dataset(cls, path: str = None):
        """
        Load the dataset from disk

        Parameters
        ----------
        path : str
            Path to the file with the preprocessed data (numpy format)
        """
        if not path:
            path = os.path.join(_age_dataset_dir, 'age_preprocessed.npy')
        cls.DATASET = np.load(path, mmap_mode='r')

    def __init__(self, min_age: int, max_age: int):
        if self.DATASET is None:
            self.load_dataset()
        mask = (self.DATASET['age'] >= min_age) & (self.DATASET['age'] < max_age)
        self.__projections = self.DATASET[mask]['projection']
        self.__age = self.DATASET[mask]['age']

    def sample(self, n: int) -> np.ndarray:
        idxs = np.random.choice(len(self.__projections), size=n)
        return self.__projections[idxs]

    def sample_with_age(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        idxs = np.random.choice(len(self.__projections), size=n)
        return self.__projections[idxs], self.__age[idxs]

    @property
    def array(self) -> np.ndarray:
        return self.__projections

    @property
    def dimensions(self) -> int:
        return self.__projections.shape[-1]
