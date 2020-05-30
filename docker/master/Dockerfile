FROM nvidia/cuda:10.1-cudnn7-devel

# Install some utilities
RUN apt-get update && \
    apt-get install -y -q wget git vim libxrender1 libsm6 bzip2 && \
    apt-get clean

# Install miniconda
RUN MINICONDA="Miniconda3-latest-Linux-x86_64.sh" && \
    wget --quiet https://repo.continuum.io/miniconda/$MINICONDA && \
    bash $MINICONDA -b -p /miniconda && \
    rm -f $MINICONDA && \
    echo ". /miniconda/etc/profile.d/conda.sh" >> ~/.bashrc
ENV PATH /miniconda/bin:$PATH

# install deepchem with master branch
RUN conda update -n base conda && \
    git clone https://github.com/deepchem/deepchem.git && \
    cd deepchem && \
    . /miniconda/etc/profile.d/conda.sh && \
    conda init bash && \
    gpu=1 bash scripts/install_deepchem_conda.sh deepchem && \
    conda activate deepchem && \
    python setup.py install && \
    conda clean -afy

RUN echo "conda activate deepchem" >> ~/.bashrc
WORKDIR /root/mydir
