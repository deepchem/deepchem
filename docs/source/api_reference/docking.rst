Docking
=======
Thanks to advances in biophysics, we are often able to find the
structure of proteins from experimental techniques like Cryo-EM or
X-ray crystallography. These structures can be powerful aides in
designing small molecules. The technique of Molecular docking performs
geometric calculations to find a "binding pose" with the small
molecule interacting with the protein in question in a suitable
binding pocket (that is, a region on the protein which has a groove in
which the small molecule can rest). For more information about
docking, check out the Autodock Vina paper:

Trott, Oleg, and Arthur J. Olson. "AutoDock Vina: improving the speed and accuracy of docking with a new scoring function, efficient optimization, and multithreading." Journal of computational chemistry 31.2 (2010): 455-461.

Binding Pocket Discovery
------------------------

DeepChem has some utilities to help find binding pockets on proteins
automatically. For now, these utilities are simple, but we will
improve these in future versions of DeepChem.

.. autoclass:: deepchem.dock.binding_pocket.BindingPocketFinder
  :members:

.. autoclass:: deepchem.dock.binding_pocket.ConvexHullPocketFinder
  :members:

Pose Generation
---------------
Pose generation is the task of finding a "pose", that is a geometric
configuration of a small molecule interacting with a protein. Pose
generation is a complex process, so for now DeepChem relies on
external software to perform pose generation. This software is invoked
and installed under the hood.

.. autoclass:: deepchem.dock.pose_generation.PoseGenerator
  :members:

.. autoclass:: deepchem.dock.pose_generation.VinaPoseGenerator
  :members:

Docking
-------
The :code:`dc.dock.docking` module provides a generic docking
implementation that depends on provide pose generation and pose
scoring utilities to perform docking. This implementation is generic.

.. autoclass:: deepchem.dock.docking.Docker
  :members:


Pose Scoring
------------
This module contains some utilities for computing docking scoring
functions directly in Python. For now, support for custom pose scoring
is limited.

.. autofunction:: deepchem.dock.pose_scoring.pairwise_distances

.. autofunction:: deepchem.dock.pose_scoring.cutoff_filter

.. autofunction:: deepchem.dock.pose_scoring.vina_nonlinearity

.. autofunction:: deepchem.dock.pose_scoring.vina_repulsion

.. autofunction:: deepchem.dock.pose_scoring.vina_hydrophobic

.. autofunction:: deepchem.dock.pose_scoring.vina_hbond

.. autofunction:: deepchem.dock.pose_scoring.vina_gaussian_first

.. autofunction:: deepchem.dock.pose_scoring.vina_gaussian_second

.. autofunction:: deepchem.dock.pose_scoring.vina_energy_term
