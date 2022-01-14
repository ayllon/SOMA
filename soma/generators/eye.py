import os.path
from typing import Dict

import numpy as np
import pandas

from soma.generators import Generator


class EyeGenerator(Generator):
    """
    Generate samples from the eye movement dataset
    Salojärvi, J., Puolamäki, K., Simola, J., Kovanen, L., Kojo, I., and Kaski, S. (2005),
    "Inferring relevance from eye movements: Feature extraction,"
    in Workshop at NIPS 2005, in Whistler, BC, Canada, on December 10, 2005.

    The dataset has been downloaded from https://www.openml.org/d/1044

    Parameters
    ----------
    label : str
        'C' = correct answer; 'R' = relevant; 'I': irrelevant
    Notes
    -----
    Columns 3 to 24 (inclusive) are features. Column 28 is the label.
    """

    LABELS = ['I', 'R', 'C']
    DATASET: pandas.DataFrame = None

    @classmethod
    def load_dataset(cls, path: str = None):
        """
        Load the dataset from disk

        Parameters
        ----------
        path : str (Optional)
            Location. It defaults to ./data/eye_movements.csv
        """
        if not path:
            path = os.path.join(os.path.dirname(__file__), 'data', 'eye_movements.csv')
        cls.DATASET = pandas.read_csv(path, usecols=list(range(2, 24)) + [27])

    @classmethod
    def count_per_label(cls) -> Dict[str, int]:
        """
        Returns
        -------
        out : dict
            A dictionary where the key is the label, and the value the number of entries for that label
        """
        if cls.DATASET is None:
            cls.load_dataset()
        counts = cls.DATASET['label'].value_counts()
        return dict([(cls.LABELS[i], counts[i]) for i in counts.index])

    def __init__(self, label: str):
        if self.DATASET is None:
            self.load_dataset()
        label_num = self.LABELS.index(label)
        self.__data = self.DATASET[self.DATASET['label'] == label_num].to_numpy()[:, 0:22]

    def sample(self, n: int) -> np.ndarray:
        idxs = np.random.choice(len(self.__data), n)
        return self.__data[idxs]

    @property
    def array(self) -> np.ndarray:
        return self.__data

    @property
    def dimensions(self) -> int:
        return self.__data.shape[1]
