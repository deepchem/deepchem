"""
Huggingface/transformers RoBERTa model for sequence-based property prediction.
"""

from numpy.lib.npyio import save
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
import os


from deepchem.models import TorchModel

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
                BPE_min_frequency = 2):

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

        super(ChemBERTaMaskedLM, self).__init__()


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
                mlm_probability: float = 0.15,
                output_dir = 'model/',
                frac_train: int = 0.95,
                eval_steps: int = 1000,
                logging_steps: int = 100,
                overwrite_output_dir: boolean = True,
                num_train_epochs: int = 10,
                per_device_train_batch_size: int = 64,
                save_steps :int = 10000,
                save_total_limit: int = 2,
                fp16: bool= True,
                run_name: str = 'chemberta_lm',
                **kwargs):
                
        # move to _fit class        
        from transformers import LineByLineTextDataset
        from transformers import DataCollatorForLanguageModeling
        from transformers import TrainingArguments

        model = ChemBERTaMaskedLM(
            vocab_size=vocab_size,
            max_position_embeddings = max_position_embeddings,
            number_attention_heads = number_attention_heads,
            num_hidden_layers = num_hidden_layers,
            dataset_path = dataset_path,
            mode = mode,
            model_path = model_path,
            model_output_dir = output_dir,
            **kwargs)

    def _fit(self):
        from transformers import LineByLineTextDataset
        from transformers import DataCollatorForLanguageModeling
        from transformers import TrainingArguments

        # move to _fit class        
        dataset = LineByLineTextDataset(file_path=self.dataset_path, 
                                            block_size=512)
        train_size = max(int(self.frac_train * len(dataset)), 1)
        eval_size = len(dataset) - train_size
        print(f"Train size: {train_size}")
        print(f"Eval size: {eval_size}")

        # move to _fit class        
        train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

        # move to _fit class        
        data_collator = DataCollatorForLanguageModeling(
                            tokenizer=self.model.tokenizer, mlm=True, 
                            mlm_probability=self.mlm_probability)

        is_gpu = torch.cuda.is_available()

        # move to _fit class
        self.training_args = TrainingArguments(
            evaluation_strategy="steps",
            eval_steps=self.eval_steps,
            load_best_model_at_end=True,
            logging_steps=self.logging_steps,
            output_dir=os.path.join(self.output_dir, self.run_name),
            overwrite_output_dir=self.overwrite_output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
            fp16 = is_gpu and self.fp16, # fp16 only works on CUDA devices
            report_to="wandb",
            run_name=self.run_name,
        )

        from transformers.trainer_callback import EarlyStoppingCallback
        from transformers import Trainer

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
        )

        trainer.train()
        trainer.save_model(os.path.join(self.output_dir, self.run_name, "final"))