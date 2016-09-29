deepchem
=============

Deep Learning Toolchain for Drug Discovery and Quantum Chemistry

Requirements
------------
* [openbabel](http://openbabel.org/wiki/Main_Page)
* [pandas](http://pandas.pydata.org/)
* [rdkit](http://www.rdkit.org/docs/Install.html)
* [boost](http://www.boost.org/)
* [joblib](https://pypi.python.org/pypi/joblib)
* [sklearn](https://github.com/scikit-learn/scikit-learn.git)
* [numpy](https://store.continuum.io/cshop/anaconda/)
* [keras](http://keras.io)
* [six](https://pypi.python.org/pypi/six)
* [dill](https://pypi.python.org/pypi/dill)
* [ipyparallel](https://ipyparallel.readthedocs.io/en/latest/)
* [mdtraj](http://mdtraj.org/)
* [tensorflow](https://www.tensorflow.org/)

Linux (64-bit) Installation from Source
---------------------------------------

```deepchem``` currently supports both Python 2.7 and Python 3.5, but is not supported on any OS'es except 64 bit linux. Please make sure you follow the directions below precisely. While you may already have system versions of some of these packages, there is no guarantee that `deepchem` will work with alternate versions than those specified below.

1. Anaconda 2.7
   Download the **64-bit Python 2.7** version of Anaconda for linux [here](https://www.continuum.io/downloads#_unix).  
   Follow the [installation instructions](http://docs.continuum.io/anaconda/install#linux-install)

2. `openbabel`
   ```bash
   conda install -c omnia openbabel
   ``` 

3. `pandas`
   ```bash
   conda install pandas 
   ```

4. `rdkit`
   ```bash
   conda install -c omnia rdkit
   ```

5. `boost`
   ```bash
   conda install -c omnia boost=1.59.0
   ```

6. `joblib`
   ```bash
   conda install joblib 
   ```

7. `keras`
   ```bash
   pip install keras --user
   ```
   `deepchem` only supports the `tensorflow` backend for keras. To set the backend to `tensorflow`,
   add the following line to your `~/.bashrc`
   ```bash
   export KERAS_BACKEND=tensorflow
   ```
   See [keras docs](https://keras.io/backend/) for more details and alternate methods of setting backend.

8. `six`
   ```bash
   conda install six
   ```
9. `dill`
    ```bash
    conda install dill
    ```

10. `ipyparallel`
    ```bash
    conda install ipyparallel
    ```

11. `mdtraj`
   ```bash
   conda install -c omnia mdtraj
   ```
   
12. `scikit-learn`
    ```bash
    conda install scikit-learn 
    ```

13. `tensorflow`: Installing `tensorflow` on older versions of Linux (which
    have glibc < 2.17) can be very challenging. For these older Linux versions,
    contact your local sysadmin to work out a custom installation. If your
    version of Linux is recent, then the following command will work:
    ```
    conda install -c https://conda.anaconda.org/jjhelmus tensorflow
    ```

14. `h5py`:
    ```
    conda install h5py
    ```

15. `deepchem`: Clone the `deepchem` github repo:
    ```bash
    git clone https://github.com/deepchem/deepchem.git
    ```
    `cd` into the `deepchem` directory and execute
    ```bash
    python setup.py install
    ```

16. To run test suite, install `nosetests`:
    ```bash
    pip install nose --user
    ```
    Make sure that the correct version of `nosetests` is active by running
    ```bash
    which nosetests 
    ```
    You might need to uninstall a system install of `nosetests` if
    there is a conflict.

17. If installation has been successful, all tests in test suite should pass:
    ```bash
    nosetests -v deepchem --nologcapture 
    ```
    Note that the full test-suite uses up a fair amount of memory. 
    Try running tests for one submodule at a time if memory proves an issue.

Frequently Asked Questions
--------------------------
1. Question: I'm seeing some failures in my test suite having to do with MKL
   ```Intel MKL FATAL ERROR: Cannot load libmkl_avx.so or libmkl_def.so.```

   Answer: This is a general issue with the newest version of `scikit-learn` enabling MKL by default. This doesn't play well with many linux systems. See BVLC/caffe#3884 for discussions. The following seems to fix the issue
   ```bash
   conda install nomkl numpy scipy scikit-learn numexpr
   conda remove mkl mkl-service
   ```
2. Question: The test suite is core-dumping for me. What's up?
   ```
   [rbharath]$ nosetests -v deepchem --nologcapture
   Illegal instruction (core dumped)
   ```
   
   Answer: This is often due to `openbabel` issues on older linux systems. Open `ipython` and run the following
   ```
   In [1]: import openbabel as ob
   ```
   If you see a core-dump, then it's a sign there's an issue with your `openbabel` install. Try reinstalling `openbabel` from source for your machine.
   
   
Getting Started
---------------
The first step to getting started is looking at the examples in the `examples/` directory. Try running some of these examples on your system and verify that the models train successfully. Afterwards, to apply `deepchem` to a new problem, try starting from one of the existing examples and modifying it step by step to work with your new use-case.
