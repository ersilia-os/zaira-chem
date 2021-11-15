from .makers import ClassificationMaker, RegressionMaker


def make_classification(n_samples, p=0.1):
    m = ClassificationMaker()
    return m.make(n_samples, p)


def make_regression(n_samples):
    m = RegressionMaker()
    return m.make(n_samples)
