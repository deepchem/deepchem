FROM nvidia/cuda:9.0-cudnn7-runtime

# Install some utilities
RUN apt-get update && \
    apt-get install -y -q wget git libxrender1 libsm6 bzip2 && \
    apt-get clean

# Install miniconda
RUN MINICONDA="Miniconda3-latest-Linux-x86_64.sh" && \
    wget --quiet https://repo.continuum.io/miniconda/$MINICONDA && \
    bash $MINICONDA -b -p /miniconda && \
    rm -f $MINICONDA
ENV PATH /miniconda/bin:$PATH

# Install deepchem conda package from omnia
# TODO: Uncomment this when there is a stable release of deepchem.
#RUN conda config --add channels omnia
#RUN conda install --yes deepchem

# Install deepchem with GPU support from github using Tue 14 Mar 2017 git head
# TODO: Get rid of this when there is a stable release of deepchem.
RUN conda update -n base conda
RUN git clone https://github.com/lilleswing/deepchem.git && \
    cd deepchem && \
    git checkout version-bumps && \
    sed -i -- 's/tensorflow$/tensorflow-gpu/g' scripts/install_deepchem_conda.sh && \
    bash scripts/install_deepchem_conda.sh && \
    python setup.py develop && \
    python -c 'import deepchem'

# Clean up
RUN cd deepchem && \
    git clean -fX

# Check that we can import DeepChem
RUN source activate deepchem
#RUN pip install nose && \
