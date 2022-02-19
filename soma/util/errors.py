from datetime import timedelta
from time import perf_counter
from typing import Callable, Dict, Optional, Tuple, Union, Type

import numpy as np
from pandas import MultiIndex, DataFrame
from tqdm.auto import tqdm

from soma.generators import Generator, Reducer
from soma.generators.kbest import KBestGenerator


def compute_errors(gen_a: Generator, gen_b: Generator, test: Callable, *, alpha: float = 0.1, samples: int = 300,
                   repeat: int = 500, return_duration: bool = False) \
        -> Union[Tuple[float, float], Tuple[float, float, Optional[timedelta]]]:
    error1 = 0.
    error2 = 0.

    duration = 0.
    for _ in range(repeat):
        a = gen_a.sample(samples)
        a2 = gen_a.sample(samples)
        b = gen_b.sample(samples)
        start = perf_counter()
        ph0 = test(a, a2)
        ph1 = test(a, b)
        end = perf_counter()
        duration += (end - start)
        error1 += ph0 <= alpha
        error2 += ph1 > alpha

    # 2 tests inside the loop!
    duration /= (2. * repeat)
    if return_duration:
        return error1 / repeat, error2 / repeat, duration
    return error1 / repeat, error2 / repeat


def stat_errors_vs_dimension(gen_a: Generator, gen_b: Generator, tests: Dict[str, Callable], *, alpha: float = 0.1,
                             samples: int = 300, repeat: int = 500, step: int = 10,
                             reducer: Type[Reducer] = KBestGenerator):
    assert gen_a.dimensions == gen_b.dimensions

    dimensions = np.unique(np.concatenate([np.arange(2, gen_a.dimensions, step), [gen_a.dimensions]]))

    kbest = reducer.fit([gen_a, gen_b])
    results = DataFrame(columns=['error1', 'error2', 'time'],
                        index=MultiIndex.from_product([tests.keys(), dimensions], names=['test', 'dimensions']))
    for d in tqdm(dimensions):
        kbest_a_gen = reducer(gen_a, d, feat_selector=kbest)
        kbest_b_gen = reducer(gen_b, d, feat_selector=kbest)
        for test_name, test in tests.items():
            results.loc[test_name, d].loc[:] = compute_errors(kbest_a_gen, kbest_b_gen, test, alpha=alpha,
                                                              samples=samples,
                                                              repeat=repeat, return_duration=True)
    return results


def stat_errors_vs_sample_size(gen_a: Generator, gen_b: Generator, tests: Dict[str, Callable], samples: np.ndarray, *,
                               alpha: float = 0.1, repeat: int = 500) -> DataFrame:
    assert gen_a.dimensions == gen_b.dimensions

    results = DataFrame(columns=['error1', 'error2', 'time'],
                        index=MultiIndex.from_product([tests.keys(), samples], names=['test', 'samples']))
    progress = tqdm(total=len(samples) * len(tests))
    for sample_size in samples:
        for test_name, test in tests.items():
            results.loc[test_name, sample_size].loc[:] = compute_errors(gen_a, gen_b, test, alpha=alpha,
                                                                        samples=sample_size,
                                                                        repeat=repeat, return_duration=True)
            progress.update(1)
    return results
