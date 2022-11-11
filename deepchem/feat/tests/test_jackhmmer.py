from deepchem.feat.sequence_featurizers.jackhmmer import Jackhmmer
from concurrent import futures
from urllib import request


test_url_pattern = 'https://storage.googleapis.com/alphafold-colab{:s}/latest/uniref90_2021_03.fasta.1'

ex = futures.ThreadPoolExecutor(3)
def fetch(source):
  request.urlretrieve(test_url_pattern.format(source))
  return source
fs = [ex.submit(fetch, source) for source in ['', '-europe', '-asia']]
source = None
for f in futures.as_completed(fs):
  source = f.result()
  ex.shutdown()
  break

# url = 'https://storage.googleapis.com/alphafold-colab-europe/latest/uniref90_2021_03.fasta.1'
j = Jackhmmer(database_path=f'https://storage.googleapis.com/alphafold-colab{source}/latest/uniref90_2021_03.fasta')
result = j.query("data/sequence.fasta")
print(result)