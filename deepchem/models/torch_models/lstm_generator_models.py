import torch.nn as nn
from torch import Tensor
from transformers import PreTrainedTokenizer, BertTokenizer
from torch.nn.utils.rnn import pad_sequence
import torch


class LSTMNeuralNet(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 num_layers: int = 1) -> None:
        super(LSTMNeuralNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers,
                           batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        output = self.fc(output)
        return output


class LSTMSampler:

    def __init__(self,
                 embedding_dim: int = 128,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 tokenizer: PreTrainedTokenizer | None = None,
                 device: str = "cpu") -> None:

        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-cased') if tokenizer is None else tokenizer
        self.embedding_dim = embedding_dim 
        self.hidden_dim = hidden_dim 
        self.num_layers = num_layers 
        self.model = None
        if device != "cpu" and not torch.cuda._is_available():
            raise ValueError("The device can't be detected other than CPU.")
        self.device = device

    def load_model_from_ckpt(self, ckpt_path: str):
        if not self.model == None:
            raise ValueError(
                "The model is already loaded for this instance. Try initiating antoher instance of the class !"
            )
        self.model = LSTMNeuralNet(self.tokenizer.vocab_size,
                                   self.embedding_dim, self.hidden_dim,
                                   self.num_layers)
        self.model.load_state_dict(torch.load(ckpt_path))
        self.model = self.model.to(self.device)

    def generate_sample(self,
                        max_len: int = 600,
                        temperature: int = 1.0) -> str:
        generated_sequence = [self.tokenizer.cls_token_id]
        for _ in range(max_len):
            input_tensor = torch.tensor(generated_sequence).unsqueeze(0)
            input_tensor = input_tensor.to(self.device)
            output = self.model(input_tensor)
            logits = output[:, -1, :] / temperature
            prbos = torch.softmax(logits, dim=-1)

            predicted_tokens = torch.multinomial(prbos, num_samples=1).item()
            generated_sequence.append(predicted_tokens)
            if predicted_tokens == self.tokenizer.sep_token_id:
                break
        generated_sequence = self.tokenizer.decode(generated_sequence,
                                                   skip_special_tokens=True)
        generated_sequence = generated_sequence.replace(" ", "")
        return generated_sequence

    def generate(self,
                 number_of_seq: int = 100,
                 max_len: int = 600,
                 temperature: int = 1.0) -> list:
        if self.model == None:
            raise ValueError(
                "The model firtst requries to load before generating samples!")
        self.model.eval()
        results = []
        for i in range(number_of_seq):
            generated_sequence = self.generate_sample(max_len=max_len,
                                                      temperature=temperature)
            results.append(generated_sequence)
        return results


class LSTMTrainer:

    def __init__(self,
                 tokenizer: PreTrainedTokenizer | None = None,
                 embedding_dim: int = 128,
                 hidden_dim: int = 256,
                 num_layers: int = 1):
        self.tokenizer = tokenizer if tokenizer else BertTokenizer.from_pretrained(
            'bert-base-cased')
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.model = LSTMNeuralNet(vocab_size=self.tokenizer.vocab_size,
                                   embedding_dim=self.embedding_dim,
                                   hidden_dim=self.hidden_dim,
                                   num_layers=self.num_layers)
        self.is_trained = False

    def prepare_dataset(self, input: list) -> Tensor:
        sequences = []
        for sequence in input:
            # Tokenize the sequence and add special tokens
            tokens = self.tokenizer.encode(sequence, add_special_tokens=True)
            sequences.append(tokens)

        # Convert the list of sequences into tensors and pad them
        padded_sequences = pad_sequence(
            [torch.tensor(seq) for seq in sequences
            ],  # Convert each sequence to a tensor
            batch_first=True,
            padding_value=self.tokenizer.
            pad_token_id  # Use BERT's padding token ID
        )

        return padded_sequences

    def train(self,
              input_sequences,
              batch_size,
              num_epochs: int,
              learning_rate: float,
              device: str = "cpu",
              criterion: any = None,
              optimizer: any = None,
              verbose: bool = False):
        criterion = criterion if criterion else nn.CrossEntropyLoss()
        optimizer = optimizer if optimizer else torch.optim.Adam(
            self.model.parameters(), lr=learning_rate)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dataset = self.prepare_dataset(input_sequences)
        criterion = criterion.to(device)
        dataset = dataset.to(device)
        self.model = self.model.to(device)
        for epoch in range(num_epochs):
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:min(i + batch_size, len(dataset))]
                # Inputs and targets
                inputs = batch[:, :-1]
                targets = batch[:, 1:]
                inputs.to(device)
                targets.to(device)

                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs.reshape(-1, self.tokenizer.vocab_size),
                                 targets.reshape(-1))

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if verbose:
                    print(
                        f'Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(dataset)}], Loss: {loss.item():.4f}'
                    )

        self.is_trained = True

    def save_model(self, save_path: str):
        if self.is_trained == False:
            raise ValueError(
                "The model is not trained yet, train the model first to save it !")
        torch.save(self.model.state_dict(), save_path)