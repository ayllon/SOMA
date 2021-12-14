import abc


class Generator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def sample(self, n: int):
        pass
