from typing import Tuple, Union

import numpy as np
import somoclu
from scipy.stats import chi2


def som_statistic(a: np.ndarray, b: np.ndarray, size: Tuple[int, int] = (10, 10), ret_som: bool = False,
                  ret_counts: bool = False, ret_df: bool = False, **kwargs):
    c = np.concatenate([a, b]).astype(np.float32)
    som = somoclu.Somoclu(size[0], size[1], **kwargs)
    som.train(c)

    # https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/chi2samp.htm
    ap = som.get_surface_state(a)
    bp = som.get_surface_state(b)

    ap = (ap <= ap.min(axis=1)[:, np.newaxis]).sum(axis=0).reshape(size[1], size[0])
    bp = (bp <= bp.min(axis=1)[:, np.newaxis]).sum(axis=0).reshape(size[1], size[0])
    gt0 = (ap + bp) > 0
    c = int(len(a) == len(b))
    c2 = np.sum(((ap - bp) ** 2)[gt0] / (ap + bp)[gt0])
    ret = [c2]
    if ret_som:
        ret.append(som)
    if ret_counts:
        ret.append((ap, bp))
    if ret_df:
        ret.append(gt0.sum() - c)
    return ret


def som_test(a: np.ndarray, b: np.ndarray, size: Tuple[int, int] = (10, 10),
             ret_som: bool = False,
             ret_counts: bool = False, **kwargs) -> Union[float, Tuple[float, ...]]:
    t, som, counts, df = som_statistic(a, b, size, ret_som=True, ret_counts=True, ret_df=True)

    ret = [1 - chi2.cdf(t, df=df)]
    if ret_som:
        ret.append(som)
    if ret_counts:
        ret.append(counts)
    if len(ret) > 1:
        return tuple(ret)
    return ret[0]
