from typing import Callable, Dict

import numpy as np
from tqdm.auto import tqdm

from soma.generators import Generator
from soma.generators.kbest import KBestGenerator


def compute_errors(gen_a: Generator, gen_b: Generator, test: Callable, *, alpha: float = 0.1, samples: int = 300,
                   repeat: int = 500):
    error1 = 0.
    error2 = 0.

    for _ in range(repeat):
        a = gen_a.sample(samples)
        a2 = gen_a.sample(samples)
        b = gen_b.sample(samples)
        ph0 = test(a, a2)
        ph1 = test(a, b)
        error1 += ph0 <= alpha
        error2 += ph1 > alpha

    return error1 / repeat, error2 / repeat


def stat_errors_vs_dimension(gen_a: Generator, gen_b: Generator, tests: Dict[str, Callable], *, alpha: float = 0.1,
                             samples: int = 300, repeat: int = 500, step: int = 10):
    assert gen_a.dimensions == gen_b.dimensions

    dimensions = np.unique(np.concatenate([np.arange(2, gen_a.dimensions, step), [gen_a.dimensions]]))

    kbest = KBestGenerator.fit([gen_a, gen_b])
    results = {test_name: np.zeros((len(dimensions), 2), dtype=float) for test_name in tests.keys()}
    for i, d in enumerate(tqdm(dimensions)):
        kbest_a_gen = KBestGenerator(gen_a, d, feat_selector=kbest)
        kbest_b_gen = KBestGenerator(gen_b, d, feat_selector=kbest)
        for test_name, test in tests.items():
            results[test_name][i, :] = compute_errors(kbest_a_gen, kbest_b_gen, test, alpha=alpha, samples=samples,
                                                      repeat=repeat)
    return dimensions, results
