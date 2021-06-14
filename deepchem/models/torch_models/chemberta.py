"""
Huggingface/transformers RoBERTa model for sequence-based property prediction.
"""

import torch.nn as nn
import torch.nn.functional as F
import os

class ChemBERTa(nn.Module):
    # defaults here match HF model: DeepChem/SmilesTokenizer_PubChem_1M.
    # TODO - figure out which to use as default, and which to pass to model class
    def __init__(self,
                vocab_size =600,
                max_position_embedddings = 515,
                number_attention_heads = 12,
                num_hidden_layers = 6,
                type_vocab_size = 1,
                dataset_path = '',
                mode = 'pre-trained',
                model_path = 'DeepChem/SmilesTokenizer_PubChem_1M',
                tokenizer_output_dir = 'tokenizer/',
                model_output_dir = 'model/',
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

        tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path, max_tokenizer_len)

        super(ChemBERTa, self).__init__(
            model, **kwargs)

