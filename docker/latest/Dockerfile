FROM nvidia/cuda:10.1-cudnn7-devel

# Install some utilities
RUN apt-get update && \
    apt-get install -y -q wget git vim libxrender1 libsm6 bzip2 && \
    apt-get clean

# Install miniconda
RUN MINICONDA="Miniconda3-latest-Linux-x86_64.sh" && \
    wget --quiet https://repo.continuum.io/miniconda/$MINICONDA && \
    bash $MINICONDA -b -p /miniconda && \
    rm -f $MINICONDA
ENV PATH /miniconda/bin:$PATH

# install latest version deepchem
RUN conda update -n base conda && \
    conda create -y --name deepchem python=3.6 && \
    . /miniconda/etc/profile.d/conda.sh && \
    conda init bash && \
    conda activate deepchem && \
    conda install -c deepchem -c rdkit -c conda-forge -c omnia deepchem-gpu
