"""
Huggingface/transformers RoBERTa model for sequence-based property prediction.
"""

import torch.nn as nn
import torch.nn.functional as F
import os

class ChemBERTaMaskedLM(nn.Module):
    # defaults here match HF model: DeepChem/SmilesTokenizer_PubChem_1M.
    # TODO - figure out which to use as default, and which to pass to model class
    def __init__(self,
                vocab_size =600,
                max_position_embedddings = 515,
                number_attention_heads = 12,
                num_hidden_layers = 6,
                type_vocab_size = 1,
                dataset_path: str = '',
                mode = 'pre-trained',
                model_path = 'DeepChem/SmilesTokenizer_PubChem_1M',
                tokenizer_output_dir = 'tokenizer/',
                tokenizer_type = 0, # if 0 - BPE, else if 1 - ST
                max_tokenizer_len = 512,
                BPE_min_frequency = 2,
                **kwargs):

        try:
            import transformers
        except:
            raise ImportError('This class requires transformers.')


        from transformers import RobertaForMaskedLM
        from transformers import RobertaConfig
        from transformers import RobertaTokenizerFast

        ''' basic flow -

        if loading pre-trained weights:
            grab pre-trained model
            grab pre-trained tokenizer (this will soon call dc.feat.RobertaFeaturizer)

        else: (from scratch training):
            generate config with suitable model params
            load model with given config
        '''

        if mode not in ['pre-trained', 'non-trained']:
            raise ValueError("mode must be either 'pre-trained' or 'non-trained'")

        super(ChemBERTa, self).__init__()


        if mode == 'pre-trained':
            self.model = RobertaForMaskedLM.from_pretrained(model_path)

            # this will be replaaced when the RobertaFeautirzer (Walid's PR) is merged.
            self.tokenizer = RobertaTokenizerFast.from_pretrained(model_path, max_len=max_tokenizer_len)

        else:
            self.config = RobertaConfig(
                vocab_size=vocab_size,
                max_position_embeddings=max_position_embedddings,
                num_attention_heads=number_attention_heads,
                num_hidden_layers=num_hidden_layers,
                type_vocab_size=type_vocab_size,
            )

            self.model = RobertaForMaskedLM(config=self.config)
            print(f"Model size: {self.model.num_parameters()} parameters.")

            if tokenizer_type == 0: # generate novel BPE tokenizer for dataset
                from transformers import ByteLevelBPETokenizer

                tokenizer_path = tokenizer_output_dir
                if not os.path.isdir(tokenizer_path):
                    os.makedirs(tokenizer_path)

                tokenizer = ByteLevelBPETokenizer()
                tokenizer.train(files=dataset_path, vocab_size=vocab_size, min_frequency=BPE_min_frequency, special_tokens=["<s>","<pad>","</s>","<unk>","<mask>"])
                tokenizer.save_model(tokenizer_path)
            else: # just assign huggingface hub model path
                tokenizer_path = model_path
                print ('load ST tokenizer')

        self.tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path, max_tokenizer_len)

class ChemBERTaMaskedLMModel(TorchModel):
    def __init__(self,
                vocab_size: int = 600,
                max_position_embeddings = 515,
                number_attention_heads=12,
                num_hidden_layers=6,
                dataset_path: str = '',
                mode='pre-trained',
                model_path='DeepChem/SmilesTokenizer_PubChem_1M',


                model_output_dir = 'model/',
                **kwargs):
        
        from transformers import LineByLineTextDataset

        model = ChemBERTa(
            vocab_size=vocab_size,
            max_position_embeddings = max_position_embeddings,
            number_attention_heads = number_attention_heads,
            num_hidden_layers = num_hidden_layers,
            dataset_path = dataset_path,
            mode = mode,
            model_path = model_path,
            model_output_dir = model_output_dir,
            **kwargs)
                
        self.dataset = LineByLineTextDataset(file_path=dataset_path, 
                                            block_size=512)

'''
ChemBERTa(vocab_size=600, max_position_embedddings=515, 
number_attention_heads=12, num_hidden_layers=6, type_vocab_size=1, 
dataset_path: str = '', mode='pre-trained', 
model_path='DeepChem/SmilesTokenizer_PubChem_1M', 
tokenizer_output_dir='tokenizer/', tokenizer_type=0, 
max_tokenizer_len=512, BPE_min_frequency=2, **kwargs)


  def __init__(self,
               n_tasks: int,
               node_out_feats: int = 64,
               edge_hidden_feats: int = 128,
               num_step_message_passing: int = 3,
               num_step_set2set: int = 6,
               num_layer_set2set: int = 3,
               mode: str = 'regression',
               number_atom_features: int = 30,
               number_bond_features: int = 11,
               n_classes: int = 2,
               self_loop: bool = False,
               **kwargs):


'''