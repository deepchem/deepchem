Installation
============

Stable version
--------------

Please install tensorflow ~2.4 before installing deepchem.

.. code-block:: bash

    pip install tensorflow~=2.4

Then, you install deepchem via pip or conda.  

.. code-block:: bash

    pip install deepchem

or 

.. code-block:: bash

    conda install -c conda-forge deepchem

RDKit is a soft requirement package, but many useful methods like molnet depend on it.
We recommend installing RDKit with deepchem if you use conda.

.. code-block:: bash

    conda install -y -c conda-forge rdkit

Nightly build version
---------------------

You install the nightly build version via pip.
The nightly version is built by the HEAD of DeepChem.

.. code-block:: bash

    pip install tensorflow~=2.4
    pip install --pre deepchem


Google Colab
------------

The fastest way to get up and running with DeepChem is to run it on
Google Colab. Check out one of the `DeepChem Tutorials`_ or this
`forum post`_ for Colab quick start guides.


Docker
------

If you want to install using a docker,
you can pull two kinds of images from `DockerHub`_.

- **deepchemio/deepchem:x.x.x**

  - Image built by using a conda (x.x.x is a version of deepchem)
  - This image is built when we push x.x.x. tag
  - Dockerfile is put in `docker/tag`_ directory

- **deepchemio/deepchem:latest**

  - Image built from source codes
  - This image is built every time we commit to the master branch
  - Dockerfile is put in `docker/nightly`_ directory

First, you pull the image you want to use.

.. code-block:: bash

    docker pull deepchemio/deepchem:latest


Then, you create a container based on the image.

.. code-block:: bash

    docker run --rm -it deepchemio/deepchem:latest

If you want GPU support:

.. code-block:: bash

    # If nvidia-docker is installed
    nvidia-docker run --rm -it deepchemio/deepchem:latest
    docker run --runtime nvidia --rm -it deepchemio/deepchem:latest

    # If nvidia-container-toolkit is installed
    docker run --gpus all --rm -it deepchemio/deepchem:latest

You are now in a docker container which deepchem was installed.
You can start playing with it in the command line.

.. code-block:: bash

    (deepchem) root@xxxxxxxxxxxxx:~/mydir# python
    Python 3.6.10 |Anaconda, Inc.| (default, May  8 2020, 02:54:21)
    [GCC 7.3.0] on linux
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import deepchem as dc

If you want to check the tox21 benchmark:

.. code-block:: bash

    # you can run our tox21 benchmark
    (deepchem) root@xxxxxxxxxxxxx:~/mydir# wget https://raw.githubusercontent.com/deepchem/deepchem/master/examples/benchmark.py
    (deepchem) root@xxxxxxxxxxxxx:~/mydir# python benchmark.py -d tox21 -m graphconv -s random


From source with conda
----------------------

**Installing via these steps will ensure you are installing from the source**.

**Prerequisite**

- Shell: Bash, Zsh, PowerShell
- Conda: >4.6


First, please clone the deepchem repository from GitHub.

.. code-block:: bash

    git clone https://github.com/deepchem/deepchem.git
    cd deepchem


Then, execute the shell script. The shell scripts require two arguments,
**python version** and **gpu/cpu**.

.. code-block:: bash

    source scripts/install_deepchem_conda.sh 3.7 cpu


If you want GPU support (we supports only CUDA 10.1):

.. code-block:: bash

    source scripts/install_deepchem_conda.sh 3.7 gpu


If you are using the Windows and the PowerShell:

.. code-block:: ps1

    . .\scripts\install_deepchem_conda.ps1 3.7 cpu


| Before activating deepchem environment, make sure conda has been initialized.
| Check if there is a :code:`(XXXX)` in your command line. 
| If not, use :code:`conda init <YOUR_SHELL_NAME>` to activate it, then:

.. code-block:: bash

    conda activate deepchem
    pip install -e .
    pytest -m "not slow" deepchem # optional


.. _`DeepChem Tutorials`: https://github.com/deepchem/deepchem/tree/master/examples/tutorials
.. _`forum post`: https://forum.deepchem.io/t/getting-deepchem-running-in-colab/81/7
.. _`DockerHub`: https://hub.docker.com/repository/docker/deepchemio/deepchem
.. _`docker/conda-forge`: https://github.com/deepchem/deepchem/tree/master/docker/conda-forge
.. _`docker/master`: https://github.com/deepchem/deepchem/tree/master/docker/master
