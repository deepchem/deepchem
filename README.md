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

Linux (64-bit) Installation 
---------------------------

```deepchem``` currently requires Python 2.7, and is not supported on any platforms except 64 bit linux. Please make sure you follow the directions below precisely. While you may already have system versions of some of these packages, there is no guarantee that `deepchem` will work with alternate versions than those specified below.

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
   pip install keras
   ```

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

14. `deepchem`: Clone the `deepchem` github repo:
    ```bash
    git clone https://github.com/deepchem/deepchem.git
    ```
    `cd` into the `deepchem` directory and execute
    ```bash
    python setup.py install
    ```

15. To run test suite, install `nosetests`:
    ```bash
    pip install nose 
    ```
    Make sure that the correct version of `nosetests` is active by running
    ```bash
    which nosetests 
    ```
    You might need to uninstall a system install of `nosetests` if
    there is a conflict.

16. If installation has been successful, all tests in test suite should pass:
    ```bash
    nosetests -v deepchem --nologcapture 
    ```
    Note that the full test-suite uses up a fair amount of memory. 
    Try running tests for one submodule at a time if memory proves an issue.
