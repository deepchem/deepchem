# Prototype

This repo is a prototype for the project "Hugging Face styled easy saving and loading model style".  
This prototype is an implementation of the project for two models: GraphConvModel and DTNN from the torch version.

The following are done:

- Implementation of save_pretrained in Torch_model class  
- Implementation of serialize_dict and deserialize_dict in Torch_model class  
- Implementation of serialize and deserialize in graphconvmodel.py and dtnn.py in torch_model  
- Implementation of deepchemmap  
- Unit tests are added in test_graph_conv.py and test_dtnn.py  
- Documentation of deepchemmap is added  

## Example

The following example demonstrates how the DeepChem API can be used to:

- Load the Tox21 dataset using the GraphConv featurizer  
- Initialize and train a GraphConvModel  
- Save the trained model using `save_pretrained`  
- Reload the saved model using `load_from_pretrained`  

```python
import deepchem as dc

# Load dataset
tasks, datasets, transformers = dc.molnet.load_tox21(featurizer='GraphConv')
train_dataset, valid_dataset, test_dataset = datasets

# Model parameters
n_tasks = len(tasks)
num_features = train_dataset.X[0].get_atom_features().shape[1]

# Initialize and train model
model = dc.models.torch_models.GraphConvModel(
    n_tasks,
    mode='classification',
    number_input_features=[num_features, 64]
)
model.fit(train_dataset, nb_epoch=50)

# Save the trained model
model.save_pretrained("save_graph")

# Load the saved model
model_reload = deepchem.deepchemmap.Map.load_from_pretrained("save_graph") 
```

Without the need for initilzation of the model again or specifying its parameters, the model was loaded with just a single API call.  

Prototype works in two stages - 
- Calling of save_pretrained -
   - Calls the serialize_dict from TorchModel which uses the model spcecific serialize function. This serialzes th model parameter dictionary
   - Saves the model weights and the serialized  dictionary as a json file in a directory.
- Calling of load_pretrained from DeepChemMap - 
   - Extracts the name from the json file.
   - Uses the model map to get the class of the corresponding model and use its deserialize function to get the original dictionary.
   - Initiate the model with the given dictionary
   - Load the model weights.


The DeepChemMap Map class works in three stages - 

- Load_param_dict - This converts a serialilized paramter dictionary back to its original format.
- Load_from_config - This initiates the model class from json file
- Load_frompretrained - Loads the model weights into the model class.

This prototype has also added the documentation of deepchemmap in docs.  
Through pytest, the unit tests added to it have been checked and they run successfully.  
This prototype also passed the CI tests for linting in the files changed.
This prototype also passes the CI unit tests for both the models.
