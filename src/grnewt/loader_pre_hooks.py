def classification(x, y, **kwargs):
    return x.to(**kwargs), y.to(**{k: v for k, v in kwargs.items() if k != "dtype"})


def regression(x, y, **kwargs):
    return x.to(**kwargs), y.to(**kwargs)
