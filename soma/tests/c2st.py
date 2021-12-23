import numpy as np
import pandas
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


def c2s_test(a: pandas.DataFrame, b: pandas.DataFrame, classifier='nn') -> float:
    """
    Based on
    "Revisiting Classifier Two-Sample Tests", Lopez 2016
    See page 3

    Parameters
    ----------
    a : pandas.DataFrame
        LHS data set
    b : pandas.DataFrame
        RHS data set
    classifier : str or object
        'nn' for Neural Network, 'knn' for nearest neighbor, or a classifier instance
    Returns
    -------
    p-value : float
    """
    if isinstance(classifier, str):
        if classifier == 'knn':
            classifier = KNeighborsClassifier()
        elif classifier == 'nn':
            classifier = MLPClassifier()
        else:
            raise ValueError(f'Unknown classifier {classifier}')
    assert hasattr(classifier, 'fit')
    assert hasattr(classifier, 'predict')

    x = np.concatenate([a, b])
    y = np.concatenate([np.zeros(len(a)), np.ones(len(b))])

    # Second: Shuffle and split into train/test
    x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True)

    # Third: Train
    classifier.fit(x_train, y_train)

    # Fourth: Test statistic is accuracy
    score = classifier.score(x_test, y_test)

    # Under H0, the score can be approximated by N(1/2, 1/(4|X_test|))
    mean, std = 0.5, np.sqrt(1 / (4 * len(x_test)))
    p = norm.cdf(score, mean, std)
    if p > 0.5:
        return 2 * (1 - p)
    return 2 * p
