.. _installation:

Installation
============

From Source
-----------
``deepchem`` is a python package which leans heavily on the python
scientific-computing ecosystem.  Since the current version of ``deepchem``
only supports installation from source, we need to install prerequisite
packages frrom various sources.

Anaconda Python
~~~~~~~~~~~~~~~

To facilitate ease-of-install, we strongly recommend that users install
Anaconda python. Anaconda will install a number of python scientific
packages by default, and guarantees suitable inter-op between these
packages. Note that the current release of ``deepchem`` uses python 2.7, so
make sure to Download the **64-bit Python 2.7** version of Anaconda for
linux `here <https://www.continuum.io/downloads#_unix>`_.  Follow the
`installation instructions <http://docs.continuum.io/anaconda/install#linux-install>`_.

Cheminformatics Packages
~~~~~~~~~~~~~~~~~~~~~~~~

``deepchem`` uses `rdkit <http://www.rdkit.org/docs/Install.html>`_  and
`openbabel <https://github.com/openbabel/openbabel>`_ to facilitate
manipulation of cheminformatic data. ``rdkit`` provides convenient ``Mol``
objects that abstract drug-like molecules. ``openbabel`` allows for easy
conversion between the myriad data-formats of chemical data. Installing
these packages from source can be challenging, so we recommend installing
with ``conda``

.. code-block:: bash

    $ conda install -c omnia rdkit
    $ conda install -c omnia openbabel

Machine Learning Packages
~~~~~~~~~~~~~~~~~~~~~~~~~
``deepchem`` uses a variety of backend machine learning
packages. Anaconda python will install ``scikit-learn`` and ``theano`` by
default. ``scikit-learn`` wraps a variety of standard machine learning
models in a convenient python API, while ``theano`` is a tensor
manipulation library.  Many models in ``deepchem`` use [keras](keras.io), a
convenient deep-learning wrapper for Theano. ``deepchem`` depends on additions
to ``keras`` which have not been merged into the master ``keras`` source,
so we need to install a custom version of ``keras`` from source.

.. code-block:: bash

    $ git clone https://github.com/pandegroup/keras.git
    $ cd keras/
    $ python setup.py install

Installing ``deepchem`` from source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    $ git clone https://github.com/pandegroup/deepchem.git
    $ cd deepchem/
    $ python setup.py develop
