try:
  from transformers import RobertaTokenizerFast
  from deepchem.feat.reaction_featurizer import RxnFeaturizer
except ModuleNotFoundError:
  pass