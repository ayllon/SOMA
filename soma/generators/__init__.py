import abc

import numpy as np


class Generator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def sample(self, n: int) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def dimensions(self) -> int:
        pass
