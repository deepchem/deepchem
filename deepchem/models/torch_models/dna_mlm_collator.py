import torch


class DNAMLMDataCollator:
    """
    Dynamic Masked Language Modeling collator for DNA sequences.

    Applies masking at batch level (dynamic masking).
    Fully torch-based and GPU compatible.
    """

    def __init__(
        self,
        mask_token_id: int,
        vocab_size: int,
        pad_token_id: int = 0,
        mask_probability: float = 0.15,
        random_token_ratio: float = 0.1,
        keep_original_ratio: float = 0.1,
    ):
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.mask_probability = mask_probability
        self.random_token_ratio = random_token_ratio
        self.keep_original_ratio = keep_original_ratio

    def __call__(self, batch):

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        device = input_ids.device

        labels = input_ids.clone()

        # Create probability matrix
        probability_matrix = torch.full(
            labels.shape,
            self.mask_probability,
            device=device,
        )

        # Do not mask padding
        padding_mask = input_ids.eq(self.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, 0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Labels: only compute loss on masked tokens
        labels[~masked_indices] = -100

        # 80% → mask token
        replace_prob = 1.0 - self.random_token_ratio - self.keep_original_ratio
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, replace_prob, device=device)).bool()
            & masked_indices
        )
        input_ids[indices_replaced] = self.mask_token_id

        # 10% → random token
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, self.random_token_ratio / (self.random_token_ratio + self.keep_original_ratio), device=device)).bool()
            & masked_indices
            & ~indices_replaced
        )

        random_tokens = torch.randint(
            low=0,
            high=self.vocab_size,
            size=labels.shape,
            device=device,
        )

        input_ids[indices_random] = random_tokens[indices_random]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
