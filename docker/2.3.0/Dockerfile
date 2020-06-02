FROM nvidia/cuda:10.1-cudnn7-devel

# Install some utilities
RUN apt-get update && \
    apt-get install -y -q wget git vim libxrender1 libsm6 bzip2

# Install miniconda
RUN MINICONDA="Miniconda3-latest-Linux-x86_64.sh" && \
    wget --quiet https://repo.continuum.io/miniconda/$MINICONDA && \
    bash $MINICONDA -b -p /miniconda && \
    rm -f $MINICONDA && \
    echo ". /miniconda/etc/profile.d/conda.sh" >> ~/.bashrc
ENV PATH /miniconda/bin:$PATH

# install latest version deepchem
RUN conda update -n base conda && \
    conda create -y --name deepchem python=3.6 && \
    . /miniconda/etc/profile.d/conda.sh && \
    conda activate deepchem && \
    conda install -c deepchem -c rdkit -c conda-forge -c omnia deepchem-gpu==2.3.0

# clean cache and reduce image size
RUN apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /usr/local/src/* && \
    conda clean -afy && \
    rm -rf ~/.cache/pip

RUN echo "conda activate deepchem" >> ~/.bashrc
WORKDIR /root/mydir
