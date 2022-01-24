# -*- coding: utf-8 -*-
"""Tools for loading, shuffling, and batching ANI datasets

The `torchani.data.load(path)` creates an iterable of raw data,
where species are strings, and coordinates are numpy ndarrays.

You can transform this iterable by using transformations.
To do a transformation, call `it.transformation_name()`. This
will return an iterable that may be cached depending on the specific
transformation.

Available transformations are listed below:

- `species_to_indices` accepts two different kinds of arguments. It converts
    species from elements (e. g. "H", "C", "Cl", etc) into internal torchani
    indices (as returned by :class:`torchani.utils.ChemicalSymbolsToInts` or
    the ``species_to_tensor`` method of a :class:`torchani.models.BuiltinModel`
    and :class:`torchani.neurochem.Constants`), if its argument is an iterable
    of species. By default species_to_indices behaves this way, with an
    argument of ``('H', 'C', 'N', 'O', 'F', 'S', 'Cl')``  However, if its
    argument is the string "periodic_table", then elements are converted into
    atomic numbers ("periodic table indices") instead. This last option is
    meant to be used when training networks that already perform a forward pass
    of :class:`torchani.nn.SpeciesConverter` on their inputs in order to
    convert elements to internal indices, before processing the coordinates.

- `subtract_self_energies` subtracts self energies from all molecules of the
    dataset. It accepts two different kinds of arguments: You can pass a dict
    of self energies, in which case self energies are directly subtracted
    according to the key-value pairs, or a
    :class:`torchani.utils.EnergyShifter`, in which case the self energies are
    calculated by linear regression and stored inside the class in the order
    specified by species_order. By default the function orders by atomic
    number if no extra argument is provided, but a specific order may be requested.

- `remove_outliers` removes some outlier energies from the dataset if present.

- `shuffle` shuffles the provided dataset. Note that if the dataset is
    not cached (i.e. it lives in the disk and not in memory) then this method
    will cache it before shuffling. This may take time and memory depending on
    the dataset size. This method may be used before splitting into validation/training
    shuffle all molecules in the dataset, and ensure a uniform sampling from
    the initial dataset, and it can also be used during training on a cached
    dataset of batches to shuffle the batches.

- `cache` cache the result of previous transformations.
    If the input is already cached this does nothing.

- `collate` creates batches and pads the atoms of all molecules in each batch
    with dummy atoms, then converts each batch to tensor. `collate` uses a
    default padding dictionary:
    ``{'species': -1, 'coordinates': 0.0, 'forces': 0.0, 'energies': 0.0}`` for
    padding, but a custom padding dictionary can be passed as an optional
    parameter, which overrides this default padding. Note that this function
    returns a generator, it doesn't cache the result in memory.

- `pin_memory` copies the tensor to pinned (page-locked) memory so that later transfer
    to cuda devices can be done faster.

you can also use `split` to split the iterable to pieces. use `split` as:

.. code-block:: python

    it.split(ratio1, ratio2, None)

where None in the end indicate that we want to use all of the rest.

Note that orderings used in :class:`torchani.utils.ChemicalSymbolsToInts` and
:class:`torchani.nn.SpeciesConverter` should be consistent with orderings used
in `species_to_indices` and `subtract_self_energies`. To prevent confusion it
is recommended that arguments to intialize converters and arguments to these
functions all order elements *by their atomic number* (e. g. if you are working
with hydrogen, nitrogen and bromine always use ['H', 'N', 'Br'] and never ['N',
'H', 'Br'] or other variations). It is possible to specify a different custom
ordering, mainly due to backwards compatibility and to fully custom atom types,
but doing so is NOT recommended, since it is very error prone.


Example:

.. code-block:: python

    energy_shifter = torchani.utils.EnergyShifter(None)
    training, validation = torchani.data.load(dspath).subtract_self_energies(energy_shifter).species_to_indices().shuffle().split(int(0.8 * size), None)
    training = training.collate(batch_size).cache()
    validation = validation.collate(batch_size).cache()

If the above approach takes too much memory for you, you can then use dataloader
with multiprocessing to achieve comparable performance with less memory usage:

.. code-block:: python

    training, validation = torchani.data.load(dspath).subtract_self_energies(energy_shifter).species_to_indices().shuffle().split(0.8, None)
    training = torch.utils.data.DataLoader(list(training), batch_size=batch_size, collate_fn=torchani.data.collate_fn, num_workers=64)
    validation = torch.utils.data.DataLoader(list(validation), batch_size=batch_size, collate_fn=torchani.data.collate_fn, num_workers=64)
"""

from os.path import join, isfile, isdir
import os
from ._pyanitools import anidataloader
from .. import utils
import importlib
import functools
import math
import random
from collections import Counter
import numpy
import gc

PKBAR_INSTALLED = importlib.util.find_spec('pkbar') is not None  # type: ignore
if PKBAR_INSTALLED:
    import pkbar

verbose = True

PROPERTIES = ('energies',)

PADDING = {
    'species': -1,
    'coordinates': 0.0,
    'forces': 0.0,
    'energies': 0.0
}


def collate_fn(samples, padding=None):
    if padding is None:
        padding = PADDING

    return utils.stack_with_padding(samples, padding)


class IterableAdapter:
    """https://stackoverflow.com/a/39564774"""
    def __init__(self, iterable_factory, length=None):
        self.iterable_factory = iterable_factory
        self.length = length

    def __iter__(self):
        return iter(self.iterable_factory())


class IterableAdapterWithLength(IterableAdapter):

    def __init__(self, iterable_factory, length):
        super().__init__(iterable_factory)
        self.length = length

    def __len__(self):
        return self.length


class Transformations:
    """Convert one reenterable iterable to another reenterable iterable"""

    @staticmethod
    def species_to_indices(reenterable_iterable, species_order=('H', 'C', 'N', 'O', 'F', 'S', 'Cl')):
        if species_order == 'periodic_table':
            species_order = utils.PERIODIC_TABLE
        idx = {k: i for i, k in enumerate(species_order)}

        def reenterable_iterable_factory():
            for d in reenterable_iterable:
                d['species'] = numpy.array([idx[s] for s in d['species']], dtype='i8')
                yield d
        try:
            return IterableAdapterWithLength(reenterable_iterable_factory, len(reenterable_iterable))
        except TypeError:
            return IterableAdapter(reenterable_iterable_factory)

    @staticmethod
    def subtract_self_energies(reenterable_iterable, self_energies=None, species_order=None):
        intercept = 0.0
        shape_inference = False
        if isinstance(self_energies, utils.EnergyShifter):
            shape_inference = True
            shifter = self_energies
            self_energies = {}
            counts = {}
            Y = []
            for n, d in enumerate(reenterable_iterable):
                species = d['species']
                count = Counter()
                for s in species:
                    count[s] += 1
                for s, c in count.items():
                    if s not in counts:
                        counts[s] = [0] * n
                    counts[s].append(c)
                for s in counts:
                    if len(counts[s]) != n + 1:
                        counts[s].append(0)
                Y.append(d['energies'])

            # sort based on the order in periodic table by default
            if species_order is None:
                species_order = utils.PERIODIC_TABLE

            species = sorted(list(counts.keys()), key=lambda x: species_order.index(x))

            X = [counts[s] for s in species]
            if shifter.fit_intercept:
                X.append([1] * n)
            X = numpy.array(X).transpose()
            Y = numpy.array(Y)
            if Y.shape[0] == 0:
                raise RuntimeError("subtract_self_energies could not find any energies in the provided dataset.\n"
                                   "Please make sure the path provided to data.load() points to a dataset has energies and is not empty or corrupted.")
            sae, _, _, _ = numpy.linalg.lstsq(X, Y, rcond=None)
            sae_ = sae
            if shifter.fit_intercept:
                intercept = sae[-1]
                sae_ = sae[:-1]
            for s, e in zip(species, sae_):
                self_energies[s] = e
            shifter.__init__(sae, shifter.fit_intercept)
        gc.collect()

        def reenterable_iterable_factory():
            for d in reenterable_iterable:
                e = intercept
                for s in d['species']:
                    e += self_energies[s]
                d['energies'] -= e
                yield d
        if shape_inference:
            return IterableAdapterWithLength(reenterable_iterable_factory, n)
        return IterableAdapter(reenterable_iterable_factory)

    @staticmethod
    def remove_outliers(reenterable_iterable, threshold1=15.0, threshold2=8.0):
        assert 'subtract_self_energies', "Transformation remove_outliers can only run after subtract_self_energies"

        # pass 1: remove everything that has per-atom energy > threshold1
        def scaled_energy(x):
            num_atoms = len(x['species'])
            return abs(x['energies']) / math.sqrt(num_atoms)
        filtered = IterableAdapter(lambda: (x for x in reenterable_iterable if scaled_energy(x) < threshold1))

        # pass 2: compute those that are outside the mean by threshold2 * std
        n = 0
        mean = 0
        std = 0
        for m in filtered:
            n += 1
            mean += m['energies']
            std += m['energies'] ** 2
        mean /= n
        std = math.sqrt(std / n - mean ** 2)

        return IterableAdapter(lambda: filter(lambda x: abs(x['energies'] - mean) < threshold2 * std, filtered))

    @staticmethod
    def shuffle(reenterable_iterable):
        if isinstance(reenterable_iterable, list):
            list_ = reenterable_iterable
        else:
            list_ = list(reenterable_iterable)
            del reenterable_iterable
            gc.collect()
        random.shuffle(list_)
        return list_

    @staticmethod
    def cache(reenterable_iterable):
        if isinstance(reenterable_iterable, list):
            return reenterable_iterable
        ret = list(reenterable_iterable)
        del reenterable_iterable
        gc.collect()
        return ret

    @staticmethod
    def collate(reenterable_iterable, batch_size, padding=None):
        def reenterable_iterable_factory(padding=None):
            batch = []
            i = 0
            for d in reenterable_iterable:
                batch.append(d)
                i += 1
                if i == batch_size:
                    i = 0
                    yield collate_fn(batch, padding)
                    batch = []
            if len(batch) > 0:
                yield collate_fn(batch, padding)

        reenterable_iterable_factory = functools.partial(reenterable_iterable_factory,
                                                         padding)
        try:
            length = (len(reenterable_iterable) + batch_size - 1) // batch_size
            return IterableAdapterWithLength(reenterable_iterable_factory, length)
        except TypeError:
            return IterableAdapter(reenterable_iterable_factory)

    @staticmethod
    def pin_memory(reenterable_iterable):
        def reenterable_iterable_factory():
            for d in reenterable_iterable:
                yield {k: d[k].pin_memory() for k in d}
        try:
            return IterableAdapterWithLength(reenterable_iterable_factory, len(reenterable_iterable))
        except TypeError:
            return IterableAdapter(reenterable_iterable_factory)


class TransformableIterable:
    def __init__(self, wrapped_iterable, transformations=()):
        self.wrapped_iterable = wrapped_iterable
        self.transformations = transformations

    def __iter__(self):
        return iter(self.wrapped_iterable)

    def __getattr__(self, name):
        transformation = getattr(Transformations, name)

        @functools.wraps(transformation)
        def f(*args, **kwargs):
            return TransformableIterable(
                transformation(self.wrapped_iterable, *args, **kwargs),
                self.transformations + (name,))

        return f

    def split(self, *nums):
        length = len(self)
        iters = []
        self_iter = iter(self)
        for n in nums:
            list_ = []
            if n is not None:
                for _ in range(int(n * length)):
                    list_.append(next(self_iter))
            else:
                for i in self_iter:
                    list_.append(i)
            iters.append(TransformableIterable(list_, self.transformations + ('split',)))
        del self_iter
        gc.collect()
        return iters

    def __len__(self):
        return len(self.wrapped_iterable)


def load(path, additional_properties=()):
    properties = PROPERTIES + additional_properties

    def h5_files(path):
        """yield file name of all h5 files in a path"""
        if isdir(path):
            for f in os.listdir(path):
                f = join(path, f)
                yield from h5_files(f)
        elif isfile(path) and (path.endswith('.h5') or path.endswith('.hdf5')):
            yield path

    def molecules():
        for f in h5_files(path):
            anidata = anidataloader(f)
            anidata_size = anidata.group_size()
            use_pbar = PKBAR_INSTALLED and verbose
            if use_pbar:
                pbar = pkbar.Pbar('=> loading {}, total molecules: {}'.format(f, anidata_size), anidata_size)
            for i, m in enumerate(anidata):
                yield m
                if use_pbar:
                    pbar.update(i)

    def conformations():
        for m in molecules():
            species = m['species']
            coordinates = m['coordinates']
            for i in range(coordinates.shape[0]):
                ret = {'species': species, 'coordinates': coordinates[i]}
                for k in properties:
                    if k in m:
                        ret[k] = m[k][i]
                yield ret

    return TransformableIterable(IterableAdapter(lambda: conformations()))


__all__ = ['load', 'collate_fn']
