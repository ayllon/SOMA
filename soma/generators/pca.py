from sklearn.decomposition import PCA

from soma.generators import Generator


class PCAGenerator(Generator):
    """
    Decorate another generator, performing a dimensionality reduction using PCA.

    Parameters
    ----------
    generator : Generator
        Generator to reduce
    dimensions : int
        Number of dimensions
    default_samples : int
        When the decorated generator has no array property (i.e. it is a true random generator,
        as MultivariateNormalGenerator), use this many samples to fit the PCA.
    """

    def __init__(self, generator: Generator, dimensions: int, default_samples: int = 10000):
        self.__pca = PCA(n_components=dimensions)
        self.__generator = generator
        if hasattr(generator, 'array'):
            self.array = self.__pca.fit_transform(generator.array)
        else:
            self.__pca.fit(generator.sample(default_samples))

    def sample(self, n: int):
        return self.__pca.transform(self.__generator.sample(n))
