# CUAEV
CUDA Extension for AEV calculation.
Performance improvement is expected to be ~3X for AEV computation and ~1.5X for energy training, 2.6X for energy+force training.

## Requirement
CUAEV needs the nightly version [pytorch](https://pytorch.org/) to be able to work.
If you use conda, you could install it by
```
conda install pytorch torchvision torchaudio cudatoolkit={YOUR_CUDA_VERSION} -c pytorch-nightly
```

## Install
In most cases, if `gcc` and `cuda` environment are well configured, runing the following command at `torchani` directory will install torchani and cuaev together.
```bash
git clone git@github.com:aiqm/torchani.git
cd torchani
# choose one option below
# use --cuaev-all-sms if you are building in SLURM environment and there are multiple different gpus in a node
# use --cuaev will only build for detected gpus
python setup.py install --cuaev          # only build for detected gpus
python setup.py install --cuaev-all-sms  # build for all gpus
# or for development
# `pip install -e . && ` is only needed for the very first install (because issue of https://github.com/pypa/pip/issues/1883)
pip install -e . && pip install -v -e . --global-option="--cuaev"          # only build for detected gpus
pip install -e . && pip install -v -e . --global-option="--cuaev-all-sms"  # build for all gpus
```


Notes for build CUAEV on multiple HPC
<details>
<summary>Bridges2</summary>

```bash
# prepare
srun -p GPU-small --ntasks=1 --cpus-per-task=5 --gpus=1 --time=02:00:00 --mem=20gb  --pty -u bash -i
module load cuda/10.2.0
conda create -n cuaev python=3.8
conda activate cuaev
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-nightly
# install torchani
git clone https://github.com/aiqm/torchani.git
cd torchani
pip install -e . && pip install -v -e . --global-option="--cuaev"
```

</details>

<details>
<summary>Hipergator</summary>

```bash
srun -p gpu --ntasks=1 --cpus-per-task=2 --gpus=geforce:1 --time=02:00:00 --mem=10gb  --pty -u bash -i
# create env if necessary
conda create -n cuaev python=3.8
conda activate cuaev
# modules
module load cuda/11.1.0 gcc/7.3.0 git
# pytorch
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-nightly -c conda-forge
# install torchani
git clone https://github.com/aiqm/torchani.git
cd torchani
pip install -e . && pip install -v -e . --global-option="--cuaev"
```

</details>

<details>
<summary>Expanse</summary>

```bash
srun -p gpu-shared --ntasks=1 --account=cwr109 --cpus-per-task=1 --gpus=1 --time=01:00:00 --mem=10gb  --pty -u bash -i
# create env if necessary
conda create -n cuaev python=3.8
conda activate cuaev
# modules
module load cuda10.2/toolkit/10.2.89 gcc/7.5.0
# pytorch
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-nightly
# install
git clone https://github.com/aiqm/torchani.git
cd torchani
pip install -e . && pip install -v -e . --global-option="--cuaev"
```

</details>


<details>
<summary>Moria</summary>

```bash
srun --ntasks=1 --cpus-per-task=2 --gpus=1 --time=02:00:00 --mem=10gb  --pty -u bash -i
# create env if necessary
conda create -n cuaev python=3.8
conda activate cuaev
# cuda path (could be added to ~/.bashrc)
export PATH=/usr/local/cuda/bin:$PATH  # nvcc for cuda 9.2
# pytorch
conda install pytorch torchvision cudatoolkit=9.2 -c pytorch-nightly
# install
git clone https://github.com/aiqm/torchani.git
cd torchani
pip install -e . && pip install -v -e . --global-option="--cuaev"
```

</details>

## Test
```bash
cd torchani
./download.sh
python tests/test_cuaev.py
```

## Usage
Pass `use_cuda_extension=True` when construct aev_computer, for example:
```python
cuaev_computer = torchani.AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species, use_cuda_extension=True)
```

## TODOs
- [x] CUAEV Forward
- [x] CUAEV Backwad (Force)
- [x] CUAEV Double Backwad (Force training need aev's double backward w.r.t. grad_aev)
- [ ] PBC

## Benchmark
Benchmark of [torchani/tools/training-aev-benchmark.py](https://github.com/aiqm/torchani/blob/master/tools/training-aev-benchmark.py):

Train ANI-1x dataset (Batchsize 2560) on Tesla V100 for 1 epoch:
```
RUN                Total AEV    Forward      Backward     Force        Optimizer    Others       Epoch time   GPU
0 cu Energy        3.355 sec    4.470 sec    4.685 sec    0.0 ms       3.508 sec    2.223 sec    18.241 sec   2780.8MB
1 py Energy        19.682 sec   4.149 sec    4.663 sec    0.0 ms       3.495 sec    2.220 sec    34.209 sec   4038.8MB
2 cu Energy+Force  3.351 sec    4.200 sec    27.402 sec   16.514 sec   3.467 sec    4.556 sec    59.490 sec   7492.8MB
3 py Energy+Force  19.964 sec   4.176 sec    91.866 sec   36.554 sec   3.473 sec    5.403 sec    161.435 sec  8034.8MB
```

Train ANI-1x dataset (Batchsize 1500) on GTX 1080 for 1 epoch:
```
RUN                Total AEV    Forward      Backward     Force        Optimizer    Others       Epoch time   GPU
0 cu Energy        14.373 sec   10.870 sec   13.100 sec   0.0 ms       11.043 sec   2.913 sec    52.299 sec   1527.5MB
1 py Energy        51.545 sec   10.228 sec   13.154 sec   0.0 ms       11.384 sec   2.874 sec    89.185 sec   2403.5MB
2 cu Energy+Force  14.275 sec   10.024 sec   85.423 sec   51.380 sec   7.396 sec    5.494 sec    173.992 sec  3577.5MB
3 py Energy+Force  51.305 sec   9.951 sec    271.078 sec  107.252 sec  7.835 sec    4.941 sec    452.362 sec  7307.5MB
```

benchmark
```
pip install pynvml pkbar
python tools/training-aev-benchmark.py download/dataset/ani-1x/sample.h5
python tools/aev-benchmark-size.py
```
