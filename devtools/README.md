Developer Notes / Tools
=======================

How to do a release
-------------------

### Pre-release
- Create an issue about cutting the release.

### Release
- Tag current master with new release version
- Look at github issues merged since last release
- Bump Dockerfile Version

### Post-release
- Update the docker images
  - sudo docker build -f Dockerfile .
  - sudo docker image list

  - // smoke test everything
  - nvidia-docker run -i -t \<IMAGE ID\>
  - python scripts/detect_devices.py // verify gpu is enabled
  - cd examples; python benchmark.py -d tox21

  - sudo docker tag \<IMAGE ID\> deepchemio/deepchem:latest
  - sudo docker push deepchemio/deepchem:latest

  - sudo docker tag \<IMAGE ID\> deepchemio/deepchem:<version>
  - sudo docker push deepchemio/deepchem:<version>
- Update conda installs
  - edit version in devtools/conda-recipes/deepchem/meta.yml
  - update requirements to be inline with scripts/install_deepchem_conda.sh
  - set deepchem anaconda org token
  - bash devtools/jenkins/conda_build.sh
- Post on Gitter
