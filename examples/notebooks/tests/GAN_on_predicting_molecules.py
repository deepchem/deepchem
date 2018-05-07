from nose.tools import timed
@timed(180)
def test_notebook():
  
  # coding: utf-8
  
  # Hi Karl!
  #
  # Last Monday you gave me an interesting task, which was to deploy Generative Adversarial Networks to predict molecular properties. I would like to give you a brief summary of my recent study of the model and its application.
  #
  # #### Literature Survey on DeepChem, GANs, MoleculeNet in drug discovery
  #
  # Papers I have read so far on GANs:
  #
  # [1] Goodfellow, Ian et al.(2014). Generative Adversarial Nets. In Neutral information processing systems.
  #
  # [2] Goodfellow, Ian (2016). NIPS 2016 Tutorial: Generative Adversarial Networks. arXiv Preprint arXiv:1701.00160.
  #
  # [3] Wu, Zhenqin et al. (2017). MoleculeNet: A Benchmark for Molecular Machine Learning. arxiv.org:1703.00564.
  #
  # [4] Killoran, Nathan et al.(2017). Generating and designing DNA with deep generative models.
  #
  # [5] Benhenda, Mostapha (2017). ChemGAN challenge for drug discovery: can AI reproduce natural chemical diversity?
  #
  # [6] Chen, Hongming et al.(2018). The rise of deep learning in drug discovery. In ScienceDirect.
  #
  # [7] This site has a list of many applications of Deep Learning in biology https://github.com/hussius/deeplearning-biology
  #
  # [8] Recent articles and discussions on AI in drug discovery, BenevolentAI and Insilico Medicine.
  #
  # I also look at the notebook for the class CS231n offered at Stanford and skimmed through Wiki articles on key terms such as feed-forward neural networks, backpropagation, Simplified molecular-input line-entry system (SMILES) to review basic understanding of Neural Networks, and Leaky ReLU.
  #
  # #### Installed DeepChem on my local machine and Ran DeepChem tutorials
  #
  # I downloaded DeepChem using easy install and look into the following tutorials carefully:
  #
  # [1] Graph Convolutions For Tox21
  #
  # [2] Multitask Networks on MUV
  #
  # [3] Conditional Generative Adversarial Network
  #
  # I found a little bug with the tutorial MNIST using GAN so I opened a new issue thread on github of the group :)
  #
  # For each tutorial, I dedicated a separate notebook, copied code snippets one by one from the tutorial to my notebook and ran cells sequentially. Sometimes I modified parameters such as epoch size, batches, etc to see the affect on validation scores and output results. Functional calls from DeepChem and TensorFlow are a bit new to me so I cltr-F through the source code on github to locate where functions are defined. I familiarized myself with methods declared in GAN inherited from TensorGraph in tensorgraph/models/gan.py and Layer in tensorgraph/layers.py.
  #
  # #### Brainstormed ideas and future directions
  #
  # I chose the tutorial "Conditional GAN" as a starting point to build a simple application to predict properties of small molecules and to generate new molecular fingerprints. The idea is to feed the Discriminator with real data obtained from experiments of small molecules with some desired properties, give the Generator latent samples and train the model to produce new molecule fingerprints that carry a similar of desired properties. This idea does sound like a stripped down version of the paper "The rise of deep learning in drug discovery"(Chen et.al 2018) and "ChemGAN challenge for drug discovery: can AI reproduce natural chemical diversity?" (Benhenda 2017). If I have a chance to contribute to the DeepChem project, I would be interested in developing popular GANs model such as Objective-Reinforced Generative Adversarial Network (ORGAN), Deep Convolutional Generative Adversarial Networks (DC-GAN), and graph-GANN in the application of drug discovery (I saw your implementation of Wasserstein Generative Adversarial Networks in GAN class).
  #
  # Given the structure of a conditional GAN model, I asked myself which molecular properties I am interested in predicting, which datasets are available to train the model, how probability distribution of some interested variables is extracted from a chosen dataset, and how the model is evaluated and compared against to assess its validity. These question lead me to read some more application tutorials from github such as Predicting Binding Affinity K_i of Ligands to a Protein, Modeling Solubility, and Basic Protein-Ligand Affinity Models to search for the appropriate dataset.
  #
  # Regarding possible directions for a summer project, I am thinking about building some GAN submodels that provide the tools for researchers to implement deep learning algoorithm without the hassle. I think this goal aligns with the grand philosophy of the project DeepChem and the company. Possible tasks include adding more functions to help parse and process data into convenients forms of intput for GAN model, implementing state-of-the-art GAN models, designing more molecular-oriented API, and examining accuracy and efficiency of models. I hope to facilitate scientsits and researchers in performing GAN models to reproduce results of using Deep Learning and GAN models from journal papers.
  #
  # ####  Planning the code
  #
  # The goal of a GAN model is to learn to generate new data that carries the same interal structure as the real input data. The model is an interplay where the Discriminator learns the boundaries between classes and the Generator learns the distribution of classes. The algorithm stops iterating when the probability that a discriminator can distinguish between a real and generative data is 1/2.
  #
  # Mathematically speaking, the Discriminator tries to increase the likelihood of giving a high probability to the real data and a low probability to the generated data. Meanwhile, the Generator is optimized to increase the probability of the generated data being rated as real. By alternating the method if gradient optimization between these two networks, the GAN model will produce the data that is as realistic as the input.
  #
  # [1] Defining the problem:
  #
  # The goal is to synthesize new molecules that have desired properties using GANs. If this task is done, the model can be improved to train the generator to sample drug candidates from a database and predict drug responses and molecules which are good at treating a certain type of disease. The desired feature is the observed R^2 that suggests the ability to make ligands "drug-like". After training, it can generate molecules from a desired distribution and use the measure R^2 as refiner of output compounds. As a result, the train model predicts new compounds that should be validated with experiments on treating a certain type of disease.
  #
  # [2] Regarding the architecture of GAN, I am deploying convolutional neural networks instead of multiple layer perceptrons.
  #
  # [3] I decided to use the data from the Tutorial Basic Protein-Ligand Affinity Models and used the standard features made available with deepchem, which are are ECFP4 fingerprints, RDKit descriptors, NNScore-style bdescriptors, and hybrid binding pocket descriptors. I am currently stuck on parsing data to the get the probability distribution
  #
  # [3] Which functions should I use to generate random noise to train the discriminator on real data for n epochs? I am thinking about implementing a LeakyReLU and test the implementation with expect errors to be small
  #
  # [4] Finally, after training model, I will use the methods from mdtraj to display newly generated molecules that possess similar properties like in the data input.
  #
  #
  # I hope that you can give me more time to implement the idea, especially I find the step of parsing data and transforming real data into appropriate data for the conditional GAN model can take more time than I expected. I have three midterms on Tue, Wed, and Thurs this week but my spring break starts this Friday and I will have more time next week. I hope to showcase you my passion and willing to learn new models and write clean code.
  #
  # Best regards,
  # Lan
  
  # In[2]:
  
  
  # Obtaining dataset similarly as in the tutorial Basic
  # Protein-Ligand Affinity Models
  
  get_ipython().run_line_magic('load_ext', 'autoreload')
  get_ipython().run_line_magic('autoreload', '2')
  get_ipython().run_line_magic('pdb', 'off')
  # set DISPLAY = True when running tutorial
  DISPLAY = False
  # set PARALLELIZE to true if you want to use ipyparallel
  PARALLELIZE = False
  import warnings
  warnings.filterwarnings('ignore')
  
  import deepchem as dc
  from deepchem.utils import download_url
  
  import os
  
  download_url("https://s3-us-west-1.amazonaws.com/deepchem.io/datasets/pdbbind_core_df.csv.gz")
  data_dir = os.path.join(dc.utils.get_data_dir())
  dataset_file= os.path.join(dc.utils.get_data_dir(), "pdbbind_core_df.csv.gz")
  raw_dataset = dc.utils.save.load_from_disk(dataset_file)
  
  #print("Type of dataset is: %s" % str(type(raw_dataset)))
  #print(raw_dataset[:5])
  #print("Shape of dataset is: %s" % str(raw_dataset.shape))
  
  grid_featurizer = dc.feat.RdkitGridFeaturizer(
  voxel_width=16.0, feature_types="voxel_combined",
  voxel_feature_types=["ecfp", "splif", "hbond", "pi_stack", "cation_pi", "salt_bridge"],
  ecfp_power=5, splif_power=5, parallel=True, flatten=True)
  compound_featurizer = dc.feat.CircularFingerprint(size=128)
  PDBBIND_tasks, (train_dataset, valid_dataset, test_dataset), transformers = dc.molnet.load_pdbbind_grid()
  
  from sklearn.ensemble import RandomForestRegressor
  
  sklearn_model = RandomForestRegressor(n_estimators=100)
  model = dc.models.SklearnModel(sklearn_model)
  model.fit(train_dataset)
  
  
  from deepchem.utils.evaluate import Evaluator
  import pandas as pd
  
  metric = dc.metrics.Metric(dc.metrics.r2_score)
  
  evaluator = Evaluator(model, train_dataset, transformers)
  train_r2score = evaluator.compute_model_performance([metric])
  print("RF Train set R^2 %f" % (train_r2score["r2_score"]))
  
  evaluator = Evaluator(model, valid_dataset, transformers)
  valid_r2score = evaluator.compute_model_performance([metric])
  print("RF Valid set R^2 %f" % (valid_r2score["r2_score"]))
  
  predictions = model.predict(test_dataset)
  print(predictions)
  
  
  # In[5]:
  
  
  import deepchem as dc
  import numpy as np
  import tensorflow as tf
  
  n_classes = 4
  class_centers = np.random.uniform(-4, 4, (n_classes, 2))
  class_transforms = []
  for i in range(n_classes):
  xscale = np.random.uniform(0.5, 2)
  yscale = np.random.uniform(0.5, 2)
  angle = np.random.uniform(0, np.pi)
  m = [[xscale*np.cos(angle), -yscale*np.sin(angle)],
  [xscale*np.sin(angle), yscale*np.cos(angle)]]
  class_transforms.append(m)
  class_transforms = np.array(class_transforms)
  
  def generate_data(n_points):
  classes = np.random.randint(n_classes, size=n_points)
  r = np.random.random(n_points)
  angle = 2*np.pi*np.random.random(n_points)
  points = (r*np.array([np.cos(angle), np.sin(angle)])).T
  points = np.einsum('ijk,ik->ij', class_transforms[classes], points)
  points += class_centers[classes]
  return classes, points
  
  get_ipython().run_line_magic('matplotlib', 'inline')
  import matplotlib.pyplot as plot
  classes, points = generate_data(1000)
  plot.scatter(x=points[:,0], y=points[:,1], c=classes)
  
  import deepchem.models.tensorgraph.layers as layers
  model = dc.models.TensorGraph(learning_rate=1e-4, use_queue=False)
  
  # Inputs to the model
  
  random_in = layers.Feature(shape=(None, 10)) # Random input to the generator
  generator_classes = layers.Feature(shape=(None, n_classes)) # The classes of the generated samples
  real_data_points = layers.Feature(shape=(None, 2)) # The training samples
  real_data_classes = layers.Feature(shape=(None, n_classes)) # The classes of the training samples
  is_real = layers.Weights(shape=(None, 1)) # Flags to distinguish real from generated samples
  
  # The generator
  
  gen_in = layers.Concat([random_in, generator_classes])
  gen_dense1 = layers.Dense(30, in_layers=gen_in, activation_fn=tf.nn.relu)
  gen_dense2 = layers.Dense(30, in_layers=gen_dense1, activation_fn=tf.nn.relu)
  generator_points = layers.Dense(2, in_layers=gen_dense2)
  model.add_output(generator_points)
  
  # The discriminator
  
  all_points = layers.Concat([generator_points, real_data_points], axis=0)
  all_classes = layers.Concat([generator_classes, real_data_classes], axis=0)
  discrim_in = layers.Concat([all_points, all_classes])
  discrim_dense1 = layers.Dense(30, in_layers=discrim_in, activation_fn=tf.nn.relu)
  discrim_dense2 = layers.Dense(30, in_layers=discrim_dense1, activation_fn=tf.nn.relu)
  discrim_prob = layers.Dense(1, in_layers=discrim_dense2, activation_fn=tf.sigmoid)
  
  
  # Discriminator
  
  discrim_real_data_loss = -layers.Log(discrim_prob+1e-10) * is_real
  discrim_gen_data_loss = -layers.Log(1-discrim_prob+1e-10) * (1-is_real)
  discrim_loss = layers.ReduceMean(discrim_real_data_loss + discrim_gen_data_loss)
  discrim_submodel = model.create_submodel(layers=[discrim_dense1, discrim_dense2, discrim_prob], loss=discrim_loss)
  
  # Generator
  
  gen_loss = -layers.ReduceMean(layers.Log(discrim_prob+1e-10) * (1-is_real))
  gen_submodel = model.create_submodel(layers=[gen_dense1, gen_dense2, generator_points], loss=gen_loss)
  
  batch_size = model.batch_size
  discrim_error = []
  gen_error = []
  for step in range(20000):
  classes, points = generate_data(batch_size)
  class_flags = dc.metrics.to_one_hot(classes, n_classes)
  feed_dict={random_in: np.random.random((batch_size, 10)),
  generator_classes: class_flags,
  real_data_points: points,
  real_data_classes: class_flags,
  is_real: np.concatenate([np.zeros((batch_size,1)), np.ones((batch_size,1))])}
  discrim_error.append(model.fit_generator([feed_dict],
  submodel=discrim_submodel,
  checkpoint_interval=0))
  if step%2 == 0:
  gen_error.append(model.fit_generator([feed_dict],
  submodel=gen_submodel,
  checkpoint_interval=0))
  if step%1000 == 999:
  print(step, np.mean(discrim_error), np.mean(gen_error))
  discrim_error = []
  gen_error = []
  
  
  
