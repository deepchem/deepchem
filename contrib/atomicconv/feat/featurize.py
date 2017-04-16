from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Joseph Gomes"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import os
import sys
from subprocess import call
from atomicnet_pdbbind_datasets import load_pdbbind_fragment_coordinates

call([
    "wget", "-c",
    "http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/pdbbind_v2015.tar.gz"
])
if not os.path.exists("v2015"):
  call(["tar", "-xvzf", "pdbbind_v2015.tar.gz"])

base_dir = os.getcwd()
pdbbind_dir = os.path.join(base_dir, "v2015")
datafile = "INDEX_core_data.2013"

frag1_num_atoms = 140
frag2_num_atoms = 821
complex_num_atoms = 908
max_num_neighbors = 8
neighbor_cutoff = 12.0

pdbbind_tasks, dataset, transformers = load_pdbbind_fragment_coordinates(
    frag1_num_atoms, frag2_num_atoms, complex_num_atoms, max_num_neighbors,
    neighbor_cutoff, pdbbind_dir, base_dir, str(datafile))
