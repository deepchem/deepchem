FROM nvidia/cuda

# Install some utilities
RUN apt-get update && \
    apt-get install -y -q wget git libxrender1 libsm6 && \
    apt-get clean

# Install miniconda
RUN MINICONDA="Miniconda2-latest-Linux-x86_64.sh" && \
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
RUN git clone https://github.com/deepchem/deepchem.git && \
    cd deepchem && \
    git checkout tags/1.3.0 && \
    sed -i -- 's/tensorflow$/tensorflow-gpu/g' scripts/install_deepchem_conda.sh && \
    bash scripts/install_deepchem_conda.sh root && \
    python setup.py develop

# Clean up
RUN cd deepchem && \
    git clean -fX

# Run tests
#RUN pip install nose && \
#    nosetests -v deepchem --nologcapture
