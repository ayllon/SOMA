from typing import Tuple, Type

import numpy as np
from tqdm.auto import tqdm

from soma.generators import Generator, Reducer
from soma.tests import kl_two_samples


def compute_divergence(gen_a: Generator, gen_b: Generator, samples: int = 200, repeat: int = 50, k: int = 10) \
        -> Tuple[float, float]:
    measures = []
    for _ in range(repeat):
        a = gen_a.sample(samples)
        b = gen_b.sample(samples)
        measures.append(kl_two_samples(a, b, k=k))
    return np.mean(measures), np.std(measures)


def divergence_vs_dimension(gen_a: Generator, gen_b: Generator, reducer: Type[Reducer], samples: int = 300,
                            repeat: int = 50, step: int = 10, k: int = 10) \
        -> Tuple[np.ndarray, float, float, np.ndarray]:
    assert gen_a.dimensions == gen_b.dimensions

    dimensions = np.unique(np.concatenate([np.arange(2, gen_a.dimensions, step), [gen_a.dimensions]]))

    # Compute for the original data
    orig_mean, orig_std = compute_divergence(gen_a, gen_b, samples=samples, repeat=repeat, k=k)

    # Compute for different projections
    divergences = np.zeros((len(dimensions), 2), dtype=float)
    reducer_fitted = reducer.fit([gen_a, gen_b])
    for i, d in enumerate(tqdm(dimensions)):
        pca_a_gen = reducer(gen_a, d, reducer_fitted)
        pca_b_gen = reducer(gen_b, d, reducer_fitted)
        divergences[i, :] = compute_divergence(pca_a_gen, pca_b_gen, samples=samples, repeat=repeat, k=k)

    return dimensions, orig_mean, orig_std, divergences
