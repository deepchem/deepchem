The DeepChem Project
====================

.. raw:: html

  <embed>
    <a href="https://github.com/deepchem/deepchem">
      <img style="position: absolute; top: 0; right: 0; border: 0;" src="https://camo.githubusercontent.com/365986a132ccd6a44c23a9169022c0b5c890c387/68747470733a2f2f73332e616d617a6f6e6177732e636f6d2f6769746875622f726962626f6e732f666f726b6d655f72696768745f7265645f6161303030302e706e67" alt="Fork me on GitHub" data-canonical-src="https://s3.amazonaws.com/github/ribbons/forkme_right_red_aa0000.png">
    </a>
  </embed>


**The DeepChem project aims to democratize deep learning for science.**

What is DeepChem?
-----------------

The DeepChem project aims to build high quality tools to democratize
the use of deep learning in the sciences. The origin of DeepChem
focused on applications of deep learning to chemistry, but the project
has slowly evolved past its roots to broader applications of deep
learning to the sciences.

The core `DeepChem Repo`_ serves as a monorepo that organizes the DeepChem suite of scientific tools. 
As the project matures, smaller more focused tool will be surfaced in more targeted repos.
DeepChem is primarily developed in Python, but we are experimenting with adding support for other languages.

What are some of the things you can use DeepChem to do? Here's a few examples:

- Predict the solubility of small drug-like molecules
- Predict binding affinity for small molecule to protein targets
- Predict physical properties of simple materials
- Analyze protein structures and extract useful descriptors
- Count the number of cells in a microscopy image
- More coming soon...

We should clarify one thing up front though. DeepChem is a machine
learning library, so it gives you the tools to solve each of the
applications mentioned above yourself. DeepChem may or may not have
prebaked models which can solve these problems out of the box.

Over time, we hope to grow the set of scientific applications DeepChem
can address. This means we need lots of help! If you're a scientist
who's interested in open source, please pitch on building DeepChem.

.. _`DeepChem Repo`: https://github.com/deepchem/deepchem

Quick Start
-----------

The fastest way to get up and running with DeepChem is to run it on
Google Colab. Check out one of the `DeepChem Tutorials`_ or this
`forum post`_ for Colab quick start guides.

If you'd like to install DeepChem locally,
we recommend installing deepchem which is nightly version and RDKit.
RDKit is a soft requirement package, but many useful methods depend on it.

.. code-block:: bash

    pip install tensorflow~=2.4
    pip install --pre deepchem
    conda install -y -c conda-forge rdkit

Then open your python and try running.

.. code-block:: python

    import deepchem

.. _`DeepChem Tutorials`: https://github.com/deepchem/deepchem/tree/master/examples/tutorials
.. _`forum post`: https://forum.deepchem.io/t/getting-deepchem-running-in-colab/81/7

About Us
--------
DeepChem is managed by a team of open source contributors. Anyone is free to join and contribute! 
DeepChem has weekly developer calls. You can find `meeting minutes`_ on our `forums`_.

DeepChem developer calls are open to the public!
To listen in, please email X.Y@gmail.com, where X=bharath and Y=ramsundar to introduce yourself and ask for an invite.

.. important::
   
    | Join our `community gitter <https://gitter.im/deepchem/Lobby>`_ to discuss DeepChem.
    | Sign up for our `forums <https://forum.deepchem.io/>`_ to talk about research, development, and general questions.

.. _`meeting minutes`: https://forum.deepchem.io/search?q=Minutes%20order%3Alatest
.. _`forums`: https://forum.deepchem.io/

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Get Started

   get_started/installation
   get_started/requirements
   get_started/tutorials
   get_started/examples
   get_started/issues

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Development Guide

   development_guide/licence
   development_guide/scientists
   development_guide/coding
   development_guide/infra

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: API Reference

   api_reference/data
   api_reference/moleculenet
   api_reference/featurizers
   api_reference/splitters
   api_reference/transformers
   api_reference/models
   api_reference/layers
   api_reference/metrics
   api_reference/hyper
   api_reference/metalearning
   api_reference/rl
   api_reference/docking
   api_reference/utils
