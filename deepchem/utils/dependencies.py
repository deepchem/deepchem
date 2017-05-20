"""
This file is for lazily importing optional dependencies
"""
import importlib


def import_lazy(module_name):
  try:
    module = importlib.import_module(module_name)
    return module
  except:
    print("Warning: No %s installed on your system" % module_name)
    print("Attempting to run %s will throw runtime errors")
    return None


xgboost = import_lazy("xgboost")
pdbfixer = import_lazy("pdbfixer")
simtk = import_lazy("simtk")
mdtraj = import_lazy("mdtraj")
