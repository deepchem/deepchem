Installing DeepChem
===================

Google Colab
------------

The fastest way to get up and running with DeepChem is to run it on
Google Colab. Check out one of the `DeepChem Tutorials`_ or this
`forum post`_ for Colab quick start guides.

Conda Installation
------------------
If you'd like to install DeepChem locally, we recommend using
:code:`conda`.  If you have :code:`conda` installed, you can install
DeepChem with the one-liner

.. code-block:: bash
    conda install -y -c deepchem -c rdkit -c conda-forge -c omnia deepchem-gpu=2.3.0

Then open your python and try running.

.. code-block:: python

    import deepchem 

Pip and Docker Installation
---------------------------
We are working on improving our pip and docker installation
capabilities. We'll update our docs once we have more information on
how to do this well.

Installing from Source
----------------------

Check out our directions on Github for how to `install from source`_.

.. _`DeepChem Tutorials`: https://github.com/deepchem/deepchem/tree/master/examples/tutorials
.. _`forum post`: https://forum.deepchem.io/t/getting-deepchem-running-in-colab/81
.. _`install from source`: https://github.com/deepchem/deepchem/blob/master/README.md#linux-64-bit-installation-from-source
