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
:code:`conda` and installing RDKit with deepchem. 
RDKit is a soft requirement package, but many useful methods like
molnet depend on it.

.. code-block:: bash

    pip install tensorflow-gpu==1.14
    conda install -y -c rdkit -c conda-forge rdkit deepchem

For CPU only support instead run

.. code-block:: bash

    pip install tensorflow==1.14
    conda install -y -c rdkit -c conda-forge rdkit deepchem

Then open your python and try running.

.. code-block:: python

    import deepchem


Pip Installation
---------------------------
We are working on improving our pip installation
capabilities. We'll update our docs once we have more information on
how to do this well.


Docker Installation
------------------

If you want to install using a docker, you can pull two kinds of images from `DockerHub`_.

- **deepchemio/deepchem:x.x.x**

  - Image built by using a conda package manager (x.x.x is a version of deepchem)
  - This image is built when we push x.x.x. tag
  - Dockerfile is put in `docker/conda-forge`_ directory

- **deepchemio/deepchem:latest**

  - Image built by the master branch of deepchem source codes
  - This image is built every time we commit to the master branch
  - Dockerfile is put in `docker/master`_ directory

First, you pull the image you want to use.

.. code-block:: bash

    docker pull deepchemio/deepchem:2.3.0


Then, you create a container based on the image.

.. code-block:: bash

    docker run -it deepchemio/deepchem:2.3.0

If you want GPU support:

.. code-block:: bash

    # If nvidia-docker is installed
    nvidia-docker run -it deepchemio/deepchem:2.3.0
    docker run --runtime nvidia -it deepchemio/deepchem:2.3.0

    # If nvidia-container-toolkit is installed
    docker run --gpus all -it deepchemio/deepchem:2.3.0

You are now in a docker container whose python has deepchem installed.

.. code-block:: bash

    # you can start playing with it in the command line
    (deepchem) root@xxxxxxxxxxx:~/mydir# python
    Python 3.6.10 |Anaconda, Inc.| (default, May  8 2020, 02:54:21)
    [GCC 7.3.0] on linux
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import deepchem as dc

    # you can run our tox21 benchmark
    (deepchem) root@xxxxxxxxxxx:~/mydir# cd /deepchem/examples
    (deepchem) root@xxxxxxxxxxx:/deepchem/examples# python benchmark.py -d tox21


Installing from Source
----------------------

Check out our directions on Github for how to `install from source`_.

.. _`DeepChem Tutorials`: https://github.com/deepchem/deepchem/tree/master/examples/tutorials
.. _`forum post`: https://forum.deepchem.io/t/getting-deepchem-running-in-colab/81
.. _`DockerHub`: https://hub.docker.com/repository/docker/deepchemio/deepchem
.. _`docker/conda-forge`: https://github.com/deepchem/deepchem/tree/master/docker/conda-forge
.. _`docker/master`: https://github.com/deepchem/deepchem/tree/master/docker/master
.. _`install from source`: https://github.com/deepchem/deepchem/blob/master/README.md#install-from-source
