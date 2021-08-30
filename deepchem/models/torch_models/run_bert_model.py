from deepchem.feat.bert_tokenizer import BertFeaturizer
from transformers import BertTokenizerFast, BertModel
from deepchem.data.data_loader import FASTALoader
from deepchem.models.torch_models.hugging_face_models import BertModelWrapper
from os.path import join, realpath, dirname
from icecream import ic

tokenizer = BertTokenizerFast.from_pretrained(
    "Rostlab/prot_bert", do_lower_case=False)
featurizer = BertFeaturizer(tokenizer)

loader = FASTALoader(
    featurizer=featurizer, legacy=False, auto_add_annotations=True)
file_loc = realpath(__file__)
directory = dirname(file_loc)
data = loader.create_dataset(
    input_files=join(directory,
                     "../../feat/tests/data/uniprot_truncated.fasta"))
print(f"data: {data}")

for datapt in data.itersamples():
  print(list(datapt))

"""
sequence_long = ['[CLS] D L I P T S S K L V V K K A F F A L V T [SEP]']
encoded_input = tokenizer(sequence_long, return_tensors='pt')
encoded_input = {
    'input_ids': encoded_input[0],
    'token_type_ids': encoded_input[1],
    'attention_mask': encoded_input[2]
}
ic(encoded_input)

model = BertModel.from_pretrained("Rostlab/prot_bert")
model = BertModelWrapper(model)
output = model(**encoded_input)
# output = model.fit(**data)
ic(f"output: {output}")
"""
