import numpy as np
from keras.datasets import mnist

from soma.generators import Generator


class NmistGenerator(Generator):
    """
    Generate samples from the distribution of values from the NMIST dataset

    Parameters
    ----------
    digit : int
        Digit from which samples must be generated
    """

    def __init__(self, digit: int):
        (train_data, train_label), (test_data, test_label) = mnist.load_data()
        data, label = np.concatenate([train_data, test_data]), np.concatenate([train_label, test_label])
        data = data.astype(float)
        self.__data = data[label == digit].reshape(-1, 28 * 28)

    @property
    def array(self) -> np.ndarray:
        return self.__data

    @property
    def dimensions(self) -> int:
        return self.__data.shape[1]

    def sample(self, n: int) -> np.ndarray:
        idxs = np.random.choice(len(self.__data), n)
        return self.__data[idxs]
