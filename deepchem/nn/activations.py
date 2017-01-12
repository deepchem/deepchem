#from __future__ import absolute_import
from keras import backend as K

def get_from_module(identifier, module_params, module_name,
                    instantiate=False, kwargs=None):
    """Retrieves a class of function member of a module.

    # Arguments
        identifier: the object to retrieve. It could be specified
            by name (as a string), or by dict. In any other case,
            `identifier` itself will be returned without any changes.
        module_params: the members of a module
            (e.g. the output of `globals()`).
        module_name: string; the name of the target module. Only used
            to format error messages.
        instantiate: whether to instantiate the returned object
            (if it's a class).
        kwargs: a dictionary of keyword arguments to pass to the
            class constructor if `instantiate` is `True`.

    # Returns
        The target object.

    # Raises
        ValueError: if the identifier cannot be found.
    """
    if isinstance(identifier, str) or isinstance(identifier, unicode):
        res = module_params.get(identifier)
        if not res:
            raise ValueError('Invalid ' + str(module_name) + ': ' +
                             str(identifier))
        if instantiate and not kwargs:
            return res()
        elif instantiate and kwargs:
            return res(**kwargs)
        else:
            return res
    elif isinstance(identifier, dict):
        name = identifier.pop('name')
        res = module_params.get(name)
        if res:
            return res(**identifier)
        else:
            raise ValueError('Invalid ' + str(module_name) + ': ' +
                             str(identifier))
    return identifier

def softmax(x):
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim == 3:
        e = K.exp(x - K.max(x, axis=-1, keepdims=True))
        s = K.sum(e, axis=-1, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor '
                         'that is not 2D or 3D. '
                         'Here, ndim=' + str(ndim))


def elu(x, alpha=1.0):
    return K.elu(x, alpha)


def softplus(x):
    return K.softplus(x)


def softsign(x):
    return K.softsign(x)


def relu(x, alpha=0., max_value=None):
    return K.relu(x, alpha=alpha, max_value=max_value)


def tanh(x):
    return K.tanh(x)


def sigmoid(x):
    return K.sigmoid(x)


def hard_sigmoid(x):
    return K.hard_sigmoid(x)


def linear(x):
    return x


def get(identifier):
    if identifier is None:
        return linear
    return get_from_module(identifier, globals(), 'activation function')
