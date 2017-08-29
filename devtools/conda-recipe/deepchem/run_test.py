import os

package_name = os.environ['package_name']
if package_name == 'deepchem' or package_name == 'deepchem-gpu':
  import deepchem
