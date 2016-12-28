"""
Generates protein-ligand docked poses using Autodock Vina.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "GPL"

import os
from subprocess import call

class VinaPoseGenerator(object):

  def __init__(self):
    """Initializes Vina Pose generation"""
    current_dir = os.path.dirname(os.path.realpath(__file__))
    self.vina_dir = os.path.join(current_dir, "autodock_vina_1_1_2_linux_x86")
    if not os.path.exists(self.vina_dir):
      print("Vina not available. Downloading")
      wget_cmd = "wget http://vina.scripps.edu/download/autodock_vina_1_1_2_linux_x86.tgz"
      call(wget_cmd.split())
      print("Downloaded Vina. Extracting")
      download_cmd = "tar xzvf autodock_vina_1_1_2_linux_x86.tgz"
      call(download_cmd.split())
      print("Moving to final location")
      mv_cmd = "mv autodock_vina_1_1_2_linux_x86 %s" % current_dir
      call(mv_cmd.split())
      print("Cleanup: removing downloaded vina tar.gz")
      rm_cmd = "rm autodock_vina_1_1_2_linux_x86.tgz"
      call(rm_cmd.split())
    self.vina_cmd = os.path.join(self.vina_dir, "bin/vina")
      

  def generate_poses(self, pdb_file, ligand_file):
    """Generates the docked complex and outputs the file."""
    pass
