import torch.nn as nn
from torch import Tensor
from transformers import PreTrainedTokenizer, BertTokenizer
from torch.nn.utils.rnn import pad_sequence
from deepchem.models.losses import Loss
from deepchem.models.torch_models import TorchModel
from deepchem.models.optimizers import Optimizer, Adam
from deepchem.models.losses import Loss, SparseSoftmaxCrossEntropy
from deepchem.utils.typing import LossFn
import torch
from typing import Iterator, Optional, Union


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


class LSTMGenerator(TorchModel):

    def __init__(self,
                 loss: Loss = SparseSoftmaxCrossEntropy(),
                 batch_size: int = 8,
                 embedding_dim: int = 128,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 tokenizer: PreTrainedTokenizer | None = None,
                 learning_rate: Union[float] = 0.001,
                 optimizer: Optional[Optimizer] = None,
                 model_dir: str = "lstm_generator_model",
                 device: str = "cpu") -> None:
        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-cased') if tokenizer is None else tokenizer
        model = LSTMNeuralNet(vocab_size=self.tokenizer.vocab_size,
                              embedding_dim=embedding_dim,
                              hidden_dim=hidden_dim,
                              num_layers=num_layers)
        model = model.to(device)
        optimizer = optimizer if optimizer else Adam(
            learning_rate=learning_rate)._create_pytorch_optimizer(model.parameters())
        super(LSTMGenerator, self).__init__(model=model,
                                            loss=loss._create_pytorch_loss(),
                                            batch_size=batch_size,
                                            model_dir=model_dir,
                                            optimizer=optimizer,
                                            device=device,
                                            learning_rate=learning_rate)

    def default_generator(self,
                          input: list,
                          num_epochs: int = 1) -> Iterator:
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

        for epoch in range(num_epochs):
            for i in range(0, len(padded_sequences), self.batch_size):
                batch = padded_sequences[i:min(i +
                                      self.batch_size, len(padded_sequences))]
                inputs = batch[:, :-1]
                targets = batch[:, 1:]
                inputs.to(self.device)
                targets.to(self.device)
                yield inputs, targets

    def fit(self, input: list, verbose: bool = False):

        return self.fit_generator(self.default_generator(input),
                                  verbose)

    def fit_generator(self,
                      generator: Iterator,
                      checkpoint_interval: int = 1000,
                      max_checkpoints_to_keep: int = 5,
                      verbose: bool = False):
        current_step = 1 
        loss_float = 0.0
        least_loss = float('inf')
        for inputs, targets in generator:
            outputs = self.model(inputs)
            # loss_val = self.loss(outputs.reshape(-1, self.tokenizer.vocab_size),
            #                      targets.reshape(-1))
            loss_val = nn.CrossEntropyLoss()(outputs.reshape(-1, self.tokenizer.vocab_size),
                                 targets.reshape(-1))
            # Backward and optimize
            self.optimizer.zero_grad()
            loss_val.backward()
            self.optimizer.step()

            if checkpoint_interval > 0 and current_step % checkpoint_interval == checkpoint_interval - 1:
                self.save_checkpoint(max_checkpoints_to_keep)
            loss_float += loss_val.item()
            least_loss = min(least_loss, loss_val.item())
            if verbose:
                print(
                    f'Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(dataset)}], Loss: {loss.item():.4f}'
                )
            current_step += 1

        avg_loss = loss_float / current_step
        return avg_loss, least_loss

    def restore(self,
                checkpoint: Optional[str] = None,
                model_dir: Optional[str] = None,
                strict: Optional[bool] = True) -> None:
        """Reload the values of all variables from a checkpoint file.

        Parameters
        ----------
        checkpoint: str
            the path to the checkpoint file to load.  If this is None, the most recent
            checkpoint will be chosen automatically.  Call get_checkpoints() to get a
            list of all available checkpoints.
        model_dir: str, default None
            Directory to restore checkpoint from. If None, use self.model_dir.  If
            checkpoint is not None, this is ignored.
        strict: bool, default True
            Whether or not to strictly enforce that the keys in checkpoint match
            the keys returned by this model's get_variable_scope() method.
        """
        if checkpoint is None:
            checkpoints = sorted(self.get_checkpoints(model_dir))
            if len(checkpoints) == 0:
                raise ValueError('No checkpoint found')
            checkpoint = checkpoints[0]
        data = torch.load(checkpoint, map_location=self.device)
        self.model = LSTMNeuralNet(self.tokenizer.vocab_size,
                                   self.embedding_dim, self.hidden_dim,
                                   self.num_layers)
        self.model.load_state_dict(data['model_state_dict'], strict=strict)

    def load_from_pretrained(self,
                             model_dir: Optional[str] = None,
                             checkpoint: Optional[str] = None) -> None:
        self.restore(model_dir=model_dir, checkpoint=checkpoint)
        self.model = self.model.to(self.device)
        self.model.eval()

    # def generate_sample(self,
    #                     max_len: int = 600,
    #                     temperature: int = 1.0) -> str:
    #     generated_sequence = [self.tokenizer.cls_token_id]
    #     for _ in range(max_len):
    #         input_tensor = torch.tensor(generated_sequence).unsqueeze(0)
    #         input_tensor = input_tensor.to(self.device)
    #         output = self.model(input_tensor)
    #         logits = output[:, -1, :] / temperature
    #         prbos = torch.softmax(logits, dim=-1)

    #         predicted_tokens = torch.multinomial(prbos, num_samples=1).item()
    #         generated_sequence.append(predicted_tokens)
    #         if predicted_tokens == self.tokenizer.sep_token_id:
    #             break
    #     generated_sequence = self.tokenizer.decode(generated_sequence,
    #                                                skip_special_tokens=True)
    #     generated_sequence = generated_sequence.replace(" ", "")
    #     return generated_sequence

    # def generate(self,
    #              number_of_seq: int = 100,
    #              max_len: int = 600,
    #              temperature: int = 1.0) -> list:
    #     if self.model == None:
    #         raise ValueError(
    #             "The model first requries to train or load before generating samples!"
    #         )
    #     self.model.eval()
    #     results = []
    #     for i in range(number_of_seq):
    #         generated_sequence = self.generate_sample(max_len=max_len,
    #                                                   temperature=temperature)
    #         results.append(generated_sequence)
    #     return results

    # def train(self,
    #           input_sequences,
    #           batch_size,
    #           num_epochs: int,
    #           learning_rate: float,
    #           device: str = "cpu",
    #           criterion: any = None,
    #           optimizer: any = None,
    #           verbose: bool = False):
    #     criterion = criterion if criterion else nn.CrossEntropyLoss()
    #     optimizer = optimizer if optimizer else torch.optim.Adam(
    #         self.model.parameters(), lr=learning_rate)
    #     if device == "auto":
    #         device = "cuda" if torch.cuda.is_available() else "cpu"
    #     dataset = self.prepare_dataset(input_sequences)
    #     criterion = criterion.to(device)
    #     dataset = dataset.to(device)
    #     self.model = self.model.to(device)
    #     for epoch in range(num_epochs):
    #         for i in range(0, len(dataset), batch_size):
    #             batch = dataset[i:min(i + batch_size, len(dataset))]
    #             # Inputs and targets
    #             inputs = batch[:, :-1]
    #             targets = batch[:, 1:]
    #             inputs.to(device)
    #             targets.to(device)

    #             # Forward pass
    #             outputs = self.model(inputs)
    #             loss = criterion(outputs.reshape(-1, self.tokenizer.vocab_size),
    #                              targets.reshape(-1))

    #             # Backward and optimize
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()

    #             if verbose:
    #                 print(
    #                     f'Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(dataset)}], Loss: {loss.item():.4f}'
    #                 )

    #     self.is_trained = True

    # def save_model(self, save_path: str):
    #     if self.is_trained == False:
    #         raise ValueError(
    #             "The model is not trained yet, train the model first to save it !"
    #         )
    #     torch.save(self.model.state_dict(), save_path)
