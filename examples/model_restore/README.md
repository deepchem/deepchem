# Model Saving/Restoration

In this example, we'll work through an example of using the
DeepChem API to save and restore a model from disk. We're going
to be training a ChemCeption model for this purpose on the
Delaney dataset. 

Here are the files we'll use

- `chemception_model.py`: The file with the model to train
- `chemception_restore.py`: The file that restores the trained model

To train the model, first run

```
python chemception_model.py
```

This will train a model and store it to a subdirectory `./model`. Let's now
invoke this model to make a prediction with it.

```
python chemception_restore.py
```

The scripts are pretty simple so go ahead and peek inside to see how they work.
