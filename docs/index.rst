.. deepchem documentation master file, created by
   sphinx-quickstart on Sat Mar  7 12:21:39 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

The DeepChem Project
====================

.. raw:: html

  <embed>
    <a href="https://github.com/deepchem/deepchem"><img style="position: absolute; top: 0; right: 0; border: 0;" src="https://camo.githubusercontent.com/365986a132ccd6a44c23a9169022c0b5c890c387/68747470733a2f2f73332e616d617a6f6e6177732e636f6d2f6769746875622f726962626f6e732f666f726b6d655f72696768745f7265645f6161303030302e706e67" alt="Fork me on GitHub" data-canonical-src="https://s3.amazonaws.com/github/ribbons/forkme_right_red_aa0000.png"></a>
  </embed>


**The DeepChem project aims to democratize deep learning for science.**

What is DeepChem?
-----------------

The DeepChem project aims to build high quality tools to democratize the use of deep learning in the sciences. The core `DeepChem Repo`_ serves as a monorepo that organizes the DeepChem suite of scientific tools. As the project matures, smaller more focused tool will be surfaced in more targeted repos.

DeepChem is primarily developed in Python, but we are experimenting with adding support for other languages.

What are some of the things you can use DeepChem to do? Here's a few examples:

- Predict the solubility of small drug-like molecules
- Predict binding affinity for small molecule to protein targets
- Predict physical properties of simple materials
- Analyze protein structures and extract useful descriptors
- Count the number of cells in a microscopy image

We should clarify one thing up front though. DeepChem is a machine
learning library, so it gives you the tools to solve each of the
applications mentioned above yourself. DeepChem may or may not have
prebaked models which can solve these problems out of the box.

Quick Start
-----------

The fastest way to get up and running with DeepChem is to run it on
Google Colab. Check out one of the `DeepChem Tutorials`_ or this
`forum post`_ for Colab quick start guides.

If you'd like to install DeepChem locally, we recommend using
:code:`conda`.  If you have :code:`conda` installed, you can install
DeepChem with the one-liner

.. code-block:: bash
    conda install -y -c deepchem -c rdkit -c conda-forge -c omnia deepchem-gpu=2.3.0

Then open your python and try running.

.. code-block:: python

    import deepchem 

.. _`DeepChem Tutorials`: https://github.com/deepchem/deepchem/tree/master/examples/tutorials
.. _`forum post`: https://forum.deepchem.io/t/getting-deepchem-running-in-colab/81

About Us
--------
DeepChem is managed by a team of open source contributors. Anyone is free to join and contribute! DeepChem has weekly developer calls. You can find `meeting minutes`_ on our `forums`_.

DeepChem developer calls are open to the public! To listen in, please email X.Y@gmail.com, where X=bharath and Y=ramsundar to introduce yourself and ask for an invite.

.. _`meeting minutes`: https://forum.deepchem.io/search?q=Minutes%20order%3Alatest
.. _`forums`: https://forum.deepchem.io/

Licensing and Commercial Uses
-----------------------------
DeepChem is licensed under the MIT License. We actively support
commercial users. Note that any novel molecules, materials, or other
discoveries powered by DeepChem belong entirely to the user and not to
DeepChem developers.

That said, we would very much appreciate a citation if you find our tools useful. You can cite DeepChem with the following reference.

.. highlight:: guess
  @book{Ramsundar-et-al-2019,
      title={Deep Learning for the Life Sciences},
      author={Bharath Ramsundar and Peter Eastman and Patrick Walters and Vijay Pande and Karl Leswing and Zhenqin Wu},
      publisher={O'Reilly Media},
      note={\url{https://www.amazon.com/Deep-Learning-Life-Sciences-Microscopy/dp/1492039837}},
      year={2019}
  }

Getting Involved
----------------

Support the DeepChem project by starring us on `on GitHub`_.  
Join our forums at https://forum.deepchem.io to participate in
discussions about research, development or any general questions. If you'd like to talk to real human beings involved in the project, say hi on our `Gitter`_ chatroom.

.. _`DeepChem repo`: https://github.com/deepchem/deepchem
.. _`on GitHub`: https://github.com/deepchem/deepchem
.. _`Gitter`: https://gitter.im/deepchem/Lobby

.. important:: Join our `community gitter <https://forms.gle/9TSdDYUgxYs8SA9e8>`_ to discuss DeepChem. Sign up for our `forums <https://forum.deepchem.io/>`_ to ralk about research, development, and general questions. 

.. toctree::
   :maxdepth: 4
   :caption: Table of Contents 
   :name: mastertoc

   Introduction <index>
   Installation <installation>
   Datasets <datasets>
   Data Loaders <dataloaders>
   Featurizers <featurizers>
   Models <models>
   Introduction to Keras <keras>
