def solver_euler_1(x, y, h, dydx, steps):
    for i in range(steps):
        y = y + h * dydx(x, y)
        x = x + h
    return y


def solver_euler_3(x, y, h, tode, steps):
    y, y1, y2 = y
    for i in range(steps):
        y_new = y + h * y1
        y1_new = y1 + h * y2
        y2_new = y2 + h * tode(x, y, y1, y2)
        x_new = x + h
        x, y, y1, y2 = x_new, y_new, y1_new, y2_new
    return x, y, y1, y2


def solver_euler_n(x, y, h, ode, steps):
    """
    Examples
    --------
    >>> import numpy as np
    >>> def tode(x, y):
    ...     y, y1, y2 = y
    ...     return np.sin(x) + np.cos(x) * y + (x ** 2) * y1 - 4 * x * y2
    >>> x_0 = 0
    >>> x_n = 2.0
    >>> y_0 = [2, -1, 3]
    >>> h = 0.1
    >>> steps = int(x_n/h)
    >>> solver_euler_n(x_0, y_0, h, tode, steps)
    (2.0000000000000004, [4.45391412194374, 2.6909199119204628, 1.1562909374702801])

    """
    for i in range(steps):
        y_new = [1] * len(y)
        for f in range(0, len(y)-1):
            y_new[f] = y[f] + h * y[f+1]
        y_new[-1] = y[-1] + h * ode(x, y)
        x_new = x + h
        x, y = x_new, y_new
    return x, y