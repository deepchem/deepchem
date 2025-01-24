import json
import tempfile
from deepchem.feat.vocabulary_builders.hf_vocab import HuggingFaceVocabularyBuilder


def testHuggingFaceVocabularyBuilder():
    from tokenizers import models, trainers
    from tokenizers.pre_tokenizers import Whitespace

    corpus = """hello world"""

    corpus_file = tempfile.NamedTemporaryFile()
    with open(corpus_file.name, 'w') as fp:
        fp.write(corpus)

    model = models.BPE(unk_token="[UNK]")
    special_tokens = ["[UNK]"]
    trainer = trainers.BpeTrainer(vocab_size=25000,
                                  special_tokens=special_tokens)

    # Build vocabulary by wrapping in huggingface vocabulary builder
    vb = HuggingFaceVocabularyBuilder(model=model, trainer=trainer)
    vb.tokenizer.pre_tokenizer = Whitespace()
    vb.build([corpus_file.name])

    vocab_file = tempfile.NamedTemporaryFile()
    vb.save(vocab_file.name)

    # Load vocabulary and do a basic sanity check on the vocabulary
    with open(vocab_file.name, 'r') as f:
        data = json.loads(f.read())
    assert len(data['added_tokens']) == 1  # [UNK]
    assert list(data['model']['vocab'].keys()) == [
        '[UNK]', 'd', 'e', 'h', 'l', 'o', 'r', 'w', 'el', 'hel', 'ld', 'lo',
        'or', 'wor', 'hello', 'world'
    ]
    print(data['model']['merges'])
    assert data['model']['merges'] == [['e', 'l'], ['h', 'el'], ['l', 'd'],
                                       ['l', 'o'], ['o', 'r'], ['w', 'or'],
                                       ['hel', 'lo'], ['wor', 'ld']]
