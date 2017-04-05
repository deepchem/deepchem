"""
This file is for lazily importing optional dependencies
"""
try:
  import xgboost as xgboost
except:
  print("Warning: No XGBOOST installed on your system")
  print("Attempting to run models with XGBOOST dependencies will throw runtime errors")
  xgboost = None