def exists(x):
    return x is not None


def default[T](x: T, val: T) -> T:
    if exists(x):
        return x
    return val