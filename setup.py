from setuptools import setup

config = {
  'install_requires': ['simdna==0.3'],
  'dependency_links': ["https://github.com/kundajelab/simdna/tarball/0.3#egg=simdna-0.3"],
  'setup_requires': ['pbr'],
  'pbr': True
}
setup(**config)
