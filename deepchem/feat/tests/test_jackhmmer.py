from deepchem.feat.sequence_featurizers.jackhmmer import Jackhmmer

j = Jackhmmer(database_path="/home/tony/github/data/uniref30")
result = j.query("data/sequence.fasta")