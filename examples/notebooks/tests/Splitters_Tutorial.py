def test_notebook():
  
  # coding: utf-8
  
  # # Using Splitters 
  
  # In this tutorial we will have a look at the various splitters that are present in deepchem library and how each of them can be used.
  
  # In[2]:
  
  
  import deepchem as dc
  import pandas as pd
  import os
  
  
  # ## Index Splitter
  
  # We start with the IndexSplitter. This splitter returns a range object which contains the split according to the fractions provided by the user. The three range objects can then be used to iterate over the dataset as test,valid and Train.
  # 
  # 
  # Each of the splitters that will be used has two functions inherited from the main class that are `train_test_split` which can be used to split the data into training and tesing data and the other fucnction is `train_valid_test_split` which is used to split the data to train, validation and test split.
  # 
  # Note: All the splitters have a default percentage of 80,10,10 as train, valid and test respectively. But can be changed by specifying the `frac_train`,`frac_test` and `frac_valid` in the ratio we want to split the data.
  
  # In[3]:
  
  
  current_dir=os.path.dirname(os.path.realpath('__file__'))
  input_data=os.path.join(current_dir,'../../deepchem/models/tests/example.csv')
  
  
  # We then featurize the data using any one of the featurizers present.
  
  # In[4]:
  
  
  tasks=['log-solubility']
  featurizer=dc.feat.CircularFingerprint(size=1024)
  loader = dc.data.CSVLoader(tasks=tasks, smiles_field="smiles",featurizer=featurizer)
  dataset=loader.featurize(input_data)
  
  
  # In[74]:
  
  
  from deepchem.splits.splitters import IndexSplitter
  
  
  # In[75]:
  
  
  splitter=IndexSplitter()
  train_data,valid_data,test_data=splitter.split(dataset)
  
  
  # In[76]:
  
  
  train_data=[i for i in train_data]
  valid_data=[i for i in valid_data]
  test_data=[i for i in test_data]
  
  
  # In[77]:
  
  
  len(train_data),len(valid_data),len(test_data)
  
  
  # As we can see that without providing the user specifications on how to split the data, the data was split into a default of 80,10,10.
  # 
  # But when we specify the parameters the dataset can be split according to our specificaitons.
  
  # In[78]:
  
  
  train_data,valid_data,test_data=splitter.split(dataset,frac_train=0.7,frac_valid=0.2,frac_test=0.1)
  
  
  # In[79]:
  
  
  train_data=[i for i in train_data]
  valid_data=[i for i in valid_data]
  test_data=[i for i in test_data]
  
  
  # In[80]:
  
  
  len(train_data),len(valid_data),len(test_data)
  
  
  # ## Specified Splitter
  
  # The next splitter that is present in the library is the specified splitter. This splitter needs a list from the dataset where it is specified which data is for training and which is for validation and testing.
  
  # In[90]:
  
  
  from deepchem.splits.splitters import SpecifiedSplitter
  current_dir=os.path.dirname(os.path.realpath('__file__'))
  input_file=os.path.join('../../deepchem/models/tests/user_specified_example.csv')
  
  tasks=['log-solubility']
  featurizer=dc.feat.CircularFingerprint(size=1024)
  loader = dc.data.CSVLoader(tasks=tasks, smiles_field="smiles",featurizer=featurizer)
  dataset=loader.featurize(input_file)
  
  split_field='split'
  
  splitter=SpecifiedSplitter(input_file,split_field)
  
  
  # In[82]:
  
  
  train_data,valid_data,test_data=splitter.split(dataset)
  
  
  # When we split the data using the specified splitter it compares the data in each row of the `split_field` which the user has to specify wether the given row should be used as training data, validation data or testing data. The user has to specify as `train`,`test` and `valid` in the `split_field`.
  # Note: The input is case insensitive.
  
  # In[83]:
  
  
  train_data,valid_data,test_data
  
  
  # ## Indice Splitter
  
  # Another splitter present in the fraework is `IndiceSplitter`. This splitter takes an input of valid_indices and test_indices which are lists with the indices of validation data and test data in the dataset respectively.
  
  # In[5]:
  
  
  from deepchem.splits.splitters import IndiceSplitter
  
  splitter=IndiceSplitter(valid_indices=[7],test_indices=[9])
  
  
  # In[6]:
  
  
  splitter.split(dataset)
  
  
  # ## RandomGroupSplitter
  
  # The splitter which can be used to split the data on the basis of groupings is the `RandomGroupSplitter`. This splitter that splits on groupings. 
  # 
  # An example use case is when there are multiple conformations of the same molecule that share the same topology.This splitter subsequently guarantees that resulting splits preserve groupings.
  # 
  # Note that it doesn't do any dynamic programming or something fancy to try to maximize the choice such that `frac_train`, `frac_valid`, or `frac_test` is maximized.It simply permutes the groups themselves. As such, use with caution if the number of elements per group varies significantly.
  # 
  # The parameter that needs to be provided with the splitter is `groups`. This is an array like list of hashables which is the same as the size of the dataset.
  # 
  
  # In[130]:
  
  
  from deepchem.splits.splitters import RandomGroupSplitter
  
  groups = [0, 4, 1, 2, 3, 7, 0, 3, 1, 0]
  solubility_dataset=dc.data.tests.load_solubility_data()
  
  
  splitter=RandomGroupSplitter(groups=groups)
  
  
  train_idxs, valid_idxs, test_idxs = splitter.split(
          solubility_dataset)
  
  
  # In[131]:
  
  
  train_idxs,valid_idxs,test_idxs
  
  
  # In[143]:
  
  
  train_data=[]
  for i in range(len(train_idxs)):
      train_data.append(groups[train_idxs[i]])
  
  valid_data=[]
  for i in range(len(valid_idxs)):
      valid_data.append(groups[valid_idxs[i]])
  
  test_data=[]
  for i in range(len(test_idxs)):
      test_data.append(groups[test_idxs[i]])
  
  
  # In[147]:
  
  
  print("Groups present in the training data =",train_data)
  print("Groups present in the validation data = ",valid_data)
  print("Groups present in the testing data = ", test_data)
  
  
  # So the `RandomGroupSplitter` when properly assigned the groups, splits the data accordingly and preserves the groupings.
  
  # ## Scaffold Splitter
  
  # The `ScaffoldSplitter` splits the data based on the scaffold of small molecules. The splitter takes the data and generates scaffolds using the smiles in the data. Then the splitter sorts the data into scaffold sets.
  
  # In[88]:
  
  
  from deepchem.splits.splitters import ScaffoldSplitter
  
  
  # In[101]:
  
  
  splitter=ScaffoldSplitter()
  solubility_dataset=dc.data.tests.load_solubility_data()
  
  
  # In[107]:
  
  
  train_data,valid_data,test_data = splitter.split(solubility_dataset,frac_train=0.7,frac_valid=0.2,frac_test=0.1)
  
  
  # In[108]:
  
  
  len(train_data),len(valid_data),len(test_data)
  
