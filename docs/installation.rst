.. _installation:

************
Installation
************

We recommend that you install ``deepchem`` with ``conda``. ::

  $ conda install -c omnia deepchem 

Conda is a cross-platform package manager built especially for scientific
python. It will install ``deepchem`` along with all dependencies from a
pre-compiled binary. If you don't have Python or the ``conda`` package
manager, we recommend starting with the `Anaconda Scientific Python
distribution <https://store.continuum.io/cshop/anaconda/>`_, which comes
pre-packaged with many of the core scientific python packages that deepchem 
uses (see below).

Supported Platforms
===================

Currently, we test and run ``deepchem`` with Python 2.7 on

- x86-64 Ubuntu 12.04 LTS, 14.04 LTS
- x86-64 CentOS 5.x, 6.x


From Source
===========
``deepchem`` is a python package which leans heavily on the python
scientific-computing ecosystem.  Since the current version of ``deepchem``
only supports installation from source, we need to install prerequisite
packages frrom various sources.

Anaconda Python
---------------

To facilitate ease-of-install, we strongly recommend that users install
Anaconda python. Anaconda will install a number of python scientific
packages by default, and guarantees suitable inter-op between these
packages. Note that the current release of ``deepchem`` uses python 2.7, so
make sure to Download the **64-bit Python 2.7** version of Anaconda for
linux `here <https://www.continuum.io/downloads#_unix>`_.  Follow the
`installation instructions <http://docs.continuum.io/anaconda/install#linux-install>`_.

Cheminformatics Packages
------------------------

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
-------------------------
``deepchem`` uses a variety of backend machine learning packages. Anaconda
python will install ``scikit-learn`` by default. ``scikit-learn`` wraps a
variety of standard machine learning models in a convenient python API.
``theano`` is a tensor manipulation library.  Many models in ``deepchem``
use [keras](keras.io), a convenient deep-learning wrapper for Theano.
``deepchem`` requires recent additions to ``theano`` and ``keras`` not yet
included in the latest releases, so we recommend installing from ``conda``.

.. code-block:: bash

    $ conda install -c omnia theano 
    $ conda install -c omnia keras 

Installing ``deepchem`` from source
-----------------------------------

.. code-block:: bash

    $ git clone https://github.com/pandegroup/deepchem.git
    $ cd deepchem/
    $ python setup.py develop

Testing the Installation
=========================
Running the tests verifies that everything is working. The test
suite uses `nose <https://nose.readthedocs.org/en/latest/>`_, which can be
installed via ``pip``. ::

  pip install nose

Then execute the command ::

  nosetests -v deepchem

