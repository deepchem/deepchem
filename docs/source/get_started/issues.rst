Known Issues & Limitations
--------------------------

Broken features
^^^^^^^^^^^^^^^

A small number of Deepchem features are known to be broken. The Deepchem team 
will either fix or deprecate these broken features. It is impossible to 
know of every possible bug in a large project like Deepchem, but we hope to 
save you some headache by listing features that we know are partially or completely 
broken.

*Note: This list is likely to be non-exhaustive. If we missed something, 
please let us know [here](https://github.com/deepchem/deepchem/issues/2376).*

+--------------------------------+-------------------+---------------------------------------------------+
| Feature                        | Deepchem response | Tracker and notes                                 |
|                                |                   |                                                   |
+================================+===================+===================================================+
| ANIFeaturizer/ANIModel         | Low Priority      | The Deepchem team recommends using TorchANI       |
|                                | Likely deprecate  | instead.                                          |
|                                |                   |                                                   |
+--------------------------------+-------------------+---------------------------------------------------+

Experimental features
^^^^^^^^^^^^^^^^^^^^^

Deepchem features usually undergo rigorous code review and testing to ensure that 
they are ready for production environments. The following Deepchem features have not 
been thoroughly tested to the level of other Deepchem modules, and could be 
potentially problematic in production environments.

*Note: This list is likely to be non-exhaustive. If we missed something, 
please let us know [here](https://github.com/deepchem/deepchem/issues/2376).*

+--------------------------------+---------------------------------------------------+
| Feature                        | Tracker and notes                                 |
|                                |                                                   |
+================================+===================================================+
| Mol2 Loading                   | Needs more testing.                               |
|                                |                                                   |
|                                |                                                   |
+--------------------------------+---------------------------------------------------+
| Interaction Fingerprints       | Needs more testing.                               |
|                                |                                                   |
|                                |                                                   |
+--------------------------------+---------------------------------------------------+

If you would like to help us address these known issues, please consider contributing to Deepchem!
