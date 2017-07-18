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
  - nvidia-docker run -i -t <IMAGE ID> // smoke test everythin

  - sudo docker tag <IMAGE ID> deepchemio/deepchem:latest
  - sudo docker push deepchemio/deepchem:latest

  - sudo docker tag <IMAGE ID> deepchemio/deepchem:<version>
  - sudo docker push deepchemio/deepchem:<version>
- Post on Gitter
