from nose.tools import timed
@timed(180)
def test_notebook():
  
  # coding: utf-8
  
  # SeqToSeq Fingerprint
  # --------------------
  # 
  # In this example, we will use a `SeqToSeq` model to generate fingerprints for classifying molecules.  This is based on the following paper, although some of the implementation details are different: Xu et al., "Seq2seq Fingerprint: An Unsupervised Deep Molecular Embedding for Drug Discovery" (https://doi.org/10.1145/3107411.3107424).
  # 
  # Many types of models require their inputs to have a fixed shape.  Since molecules can vary widely in the numbers of atoms and bonds they contain, this makes it hard to apply those models to them.  We need a way of generating a fixed length "fingerprint" for each molecule.  Various ways of doing this have been designed, such as Extended-Connectivity Fingerprints (ECFPs).  But in this example, instead of designing a fingerprint by hand, we will let a `SeqToSeq` model learn its own method of creating fingerprints.
  # 
  # A `SeqToSeq` model performs sequence to sequence translation.  For example, they are often used to translate text from one language to another.  It consists of two parts called the "encoder" and "decoder".  The encoder is a stack of recurrent layers.  The input sequence is fed into it, one token at a time, and it generates a fixed length vector called the "embedding vector".  The decoder is another stack of recurrent layers that performs the inverse operation: it takes the embedding vector as input, and generates the output sequence.  By training it on appropriately chosen input/output pairs, you can create a model that performs many sorts of transformations.
  # 
  # In this case, we will use SMILES strings describing molecules as the input sequences.  We will train the model as an autoencoder, so it tries to make the output sequences identical to the input sequences.  For that to work, the encoder must create embedding vectors that contain all information from the original sequence.  That's exactly what we want in a fingerprint, so perhaps those embedding vectors will then be useful as a way to represent molecules in other models!
  # 
  # Let's start by loading the data.  We will use the MUV dataset.  It includes 74,501 molecules in the training set, and 9313 molecules in the validation set, so it gives us plenty of SMILES strings to work with.
  
  # In[1]:
  
  
  import deepchem as dc
  tasks, datasets, transformers = dc.molnet.load_muv()
  train_dataset, valid_dataset, test_dataset = datasets
  train_smiles = train_dataset.ids
  valid_smiles = valid_dataset.ids
  
  
  # We need to define the "alphabet" for our `SeqToSeq` model, the list of all tokens that can appear in sequences.  (It's also possible for input and output sequences to have different alphabets, but since we're training it as an autoencoder, they're identical in this case.)  Make a list of every character that appears in any training sequence.
  
  # In[2]:
  
  
  tokens = set()
  for s in train_smiles:
    tokens = tokens.union(set(c for c in s))
  tokens = sorted(list(tokens))
  
  
  # Create the model and define the optimization method to use.  In this case, learning works much better if we gradually decrease the learning rate.  We use an `ExponentialDecay` to multiply the learning rate by 0.9 after each epoch.
  
  # In[3]:
  
  
  from deepchem.models.tensorgraph.optimizers import Adam, ExponentialDecay
  max_length = max(len(s) for s in train_smiles)
  model = dc.models.SeqToSeq(tokens,
                             tokens,
                             max_length,
                             encoder_layers=2,
                             decoder_layers=2,
                             embedding_dimension=256,
                             model_dir='fingerprint')
  batches_per_epoch = len(train_smiles)/model.batch_size
  model.set_optimizer(Adam(learning_rate=ExponentialDecay(0.004, 0.9, batches_per_epoch)))
  
  
  # Let's train it!  The input to `fit_sequences()` is a generator that produces input/output pairs.  On a good GPU, this should take a few hours or less.
  
  # In[4]:
  
  
  def generate_sequences(epochs):
    for i in range(epochs):
      for s in train_smiles:
        yield (s, s)
  
  model.fit_sequences(generate_sequences(1))
  
  
  # Let's see how well it works as an autoencoder.  We'll run the first 500 molecules from the validation set through it, and see how many of them are exactly reproduced.
  
  # In[5]:
  
  
  predicted = model.predict_from_sequences(valid_smiles[:500])
  count = 0
  for s,p in zip(valid_smiles[:500], predicted):
    if ''.join(p) == s:
      count += 1
  print('reproduced', count, 'of 500 validation SMILES strings')
  
  
  # Now we'll trying using the encoder as a way to generate molecular fingerprints.  We compute the embedding vectors for all molecules in the training and validation datasets, and create new datasets that have those as their feature vectors.  The amount of data is small enough that we can just store everything in memory.
  
  # In[6]:
  
  
  train_embeddings = model.predict_embeddings(train_smiles)
  train_embeddings_dataset = dc.data.NumpyDataset(train_embeddings,
                                                  train_dataset.y,
                                                  train_dataset.w,
                                                  train_dataset.ids)
  
  valid_embeddings = model.predict_embeddings(valid_smiles)
  valid_embeddings_dataset = dc.data.NumpyDataset(valid_embeddings,
                                                  valid_dataset.y,
                                                  valid_dataset.w,
                                                  valid_dataset.ids)
  
  
  # For classification, we'll use a simple fully connected network with one hidden layer.
  
  # In[7]:
  
  
  classifier = dc.models.MultiTaskClassifier(n_tasks=len(tasks),
                                                        n_features=256,
                                                        layer_sizes=[512])
  classifier.fit(train_embeddings_dataset, nb_epoch=1)
  
  
  # Find out how well it worked.  Compute the ROC AUC for the training and validation datasets.
  
  # In[8]:
  
  
  import numpy as np
  metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean, mode="classification")
  train_score = classifier.evaluate(train_embeddings_dataset, [metric], transformers)
  valid_score = classifier.evaluate(valid_embeddings_dataset, [metric], transformers)
  print('Training set ROC AUC:', train_score)
  print('Validation set ROC AUC:', valid_score)
  
