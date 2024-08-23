from typing import Dict, Any, List, Tuple
import deepchem as dc
from deepchem.models.torch_models.hf_models import HuggingFaceModel
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer
from datasets import Dataset
import numpy as np

class ESM2(HuggingFaceModel):
    """
    ESM-2 Model
    This class implements a ESM-2 model for token classification.
    This model is implemented using the HuggingFace transformers library.
    """
    def __init__(self,
                 task: str,  # type of learning task: 'token_classification'
                 model_path: str = 'facebook/esm2_t6_8M_UR50D',  # path to the HuggingFace model
                 n_labels: int = 2,  # number of classes
                 config: dict = {},  # additional configuration parameters for the model
                 **kwargs) -> None:
        self.n_labels = n_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        config = AutoConfig.from_pretrained(model_path, num_labels=n_labels, **config)
        if task == 'token_classification':
            model = AutoModelForTokenClassification.from_pretrained(model_path, config=config)
        else:
            raise ValueError('Invalid task specification')

        super(ESM2, self).__init__(model=model, task=task, tokenizer=self.tokenizer, **kwargs)

    def tokenize_sequences(self, sequences, padding=True, truncation=True, max_length=1000, return_tensors="pt", is_split_into_words=False):
        return self.tokenizer(sequences, padding=padding, truncation=truncation, max_length=max_length, return_tensors=return_tensors, is_split_into_words=is_split_into_words)
    
    def truncate_labels(self, labels, max_length):
        """Truncate labels to the specified max_length."""
        return [label[:max_length] for label in labels]
    
    def chunk_sequences(self, sequences, labels, chunk_size = 1000):
        """Split sequences and labels into chunks of size "chunk_size" or less."""
        new_sequences = []
        new_labels = []
        for seq, lbl in zip(sequences, labels):
            if len(seq) > chunk_size:
                # Split the sequence and labels into chunks of size `chunk_size` or less
                for i in range(0, len(seq), chunk_size):
                    new_sequences.append(seq[i:i+chunk_size])
                    new_labels.append(lbl[i:i+chunk_size])
            else:
                new_sequences.append(seq)
                new_labels.append(lbl)
        return new_sequences, new_labels
    
    def homogeneize_labels_length(self, labels):
        """Pad labels of shorter proteins so they all have homogeneous shape."""
        max_length_lbl = max(map(len, labels))
        new_labels = []
        for lbl in labels:
            if len(lbl) < max_length_lbl:
                new_labels.append(lbl+[2]*(max_length_lbl - len(lbl)))
            else:
                new_labels.append(lbl)
        return new_labels
        
    def prepare_dataset(self, sequences: List[str], labels: List[List[int]], chunk_size=1000, padding=True, truncation=True):
        """Prepare the dataset for training or evaluation."""
        sequences, labels = self.chunk_sequences(sequences, labels, chunk_size)
        tokenized_inputs = self.tokenize_sequences(sequences, padding, truncation, max_length=chunk_size, return_tensors="pt", is_split_into_words=False)
        labels= self.truncate_labels(labels, max_length=tokenized_inputs["input_ids"].shape[1])
        tokenized_inputs["labels"] = self.homogeneize_labels_length(labels)
        # X = np.array(tokenized_inputs)  # Use raw sequences
        # y = np.array(tokenized_inputs["labels"])
        # w = np.ones_like(y)  # Weights are set to 1
        # dataset = dc.data.NumpyDataset(X, y, w)
        dataset = dc.data.NumpyDataset(X=tokenized_inputs['input_ids'], y=np.array(tokenized_inputs["labels"]), w=np.array(tokenized_inputs['attention_mask']))
        return dataset

    def _prepare_batch(self, batch: Tuple[Any, Any, Any]):
        """Override the _prepare_batch method to handle pre-tokenized inputs."""
        sequences, y, w = batch
        tokens = self.tokenizer(sequences[0].tolist(), padding=True, return_tensors="pt", is_split_into_words=True)
        inputs = {"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"]}
        return inputs, y, w

# Example usage
if __name__ == "__main__":
    # Define the model name and number of labels
    model_name = 'facebook/esm2_t6_8M_UR50D'
    num_labels = 2  # Binary classification (binding site or not)

    # Create an instance of the custom model
    model = ESM2(task='token_classification', model_path=model_name, n_labels=num_labels)

    # Example data
    sequences = ["MKTAYIAKQRQISFVKSHFSRQDILDLWQ", "GAGVVGIVLGGFGAGVIGTLLISAGK"]
    labels = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    max_length = 10


    # sequences, labels = model.chunk_sequences(sequences, labels, chunk_size=max_length)

#TODO: Add a positive test case for the ESM2 model

    # Tokenize the sequences using the model's tokenizer
#    tokenized_sequences = model.tokenize_sequences(sequences, padding=True, truncation=True, max_length=1000, return_tensors="pt", is_split_into_words=False)

    # # Truncate the labels to match the length of the tokenized sequences
    # max_length = tokenized_sequences['input_ids'].shape[1]
    # labels = model.truncate_labels(labels, max_length)

    # # Create a dataset
    # tokenized_dict = {k: v for k, v in tokenized_sequences.items()}
    # dataset = Dataset.from_dict(tokenized_dict)
    # dataset = dataset.add_column("labels", labels)




    device = torch.device("mps") if torch.has_mps else torch.device("cpu")

    # Move model to MPS device
    model.to(device)

    # Move data to MPS device
    sequences = sequences.to(device)
    labels = labels.to(device)
    dataset = model.prepare_dataset(sequences, labels, chunk_size=10)
    dataset.X = torch.tensor(dataset.X).to(device) 
    dataset.y = torch.tensor(dataset.y).to(device)

    # Train the model (example)
    finetuning_loss = model.fit(dataset, nb_epoch=1)

    # prediction and evaluation
    result = model.predict(dataset)



# Convert to DeepChem Dataset
tokenized_sequences = model.tokenize_sequences(sequences, padding=True, truncation=True, max_length=1000, return_tensors="pt", is_split_into_words=False)
tokenized_sequences["labels"] = labels
X = np.array(tokenized_sequences["input_ids"])
y = np.array(tokenized_sequences["labels"])
dataset = dc.data.NumpyDataset(X,)
