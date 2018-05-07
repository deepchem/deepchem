from nose.tools import timed
@timed(180)
def test_notebook():
  
  # coding: utf-8
  
  # # Uncertainty in Deep Learning
  # 
  # A common criticism of deep learning models is that they tend to act as black boxes.  A model produces outputs, but doesn't given enough context to interpret them properly.  How reliable are the model's predictions?  Are some predictions more reliable than others?  If a model predicts a value of 5.372 for some quantity, should you assume the true value is between 5.371 and 5.373?  Or that it's between 2 and 8?  In some fields this situation might be good enough, but not in science.  For every value predicted by a model, we also want an estimate of the uncertainty in that value so we can know what conclusions to draw based on it.
  # 
  # DeepChem makes it very easy to estimate the uncertainty of predicted outputs (at least for the models that support it—not all of them do).  Let's start by seeing an example of how to generate uncertainty estimates.  We load a dataset, create a model, train it on the training set, and predict the output on the test set.
  
  # In[1]:
  
  
  import deepchem as dc
  import numpy as np
  import matplotlib.pyplot as plot
  
  tasks, datasets, transformers = dc.molnet.load_sampl()
  train_dataset, valid_dataset, test_dataset = datasets
  
  model = dc.models.MultiTaskRegressor(len(tasks), 1024, uncertainty=True)
  model.fit(train_dataset, nb_epoch=200)
  y_pred, y_std = model.predict_uncertainty(test_dataset)
  
  
  # All of this looks exactly like any other example, with just two differences.  First, we add the option `uncertainty=True` when creating the model.  This instructs it to add features to the model that are needed for estimating uncertainty.  Second, we call `predict_uncertainty()` instead of `predict()` to produce the output.  `y_pred` is the predicted outputs.  `y_std` is another array of the same shape, where each element is an estimate of the uncertainty (standard deviation) of the corresponding element in `y_pred`.  And that's all there is to it!  Simple, right?
  # 
  # Of course, it isn't really that simple at all.  DeepChem is doing a lot of work to come up with those uncertainties.  So now let's pull back the curtain and see what is really happening.  (For the full mathematical details of calculating uncertainty, see https://arxiv.org/abs/1703.04977)
  # 
  # To begin with, what does "uncertainty" mean?  Intuitively, it is a measure of how much we can trust the predictions.  More formally, we expect that the true value of whatever we are trying to predict should usually be within a few standard deviations of the predicted value.  But uncertainty comes from many sources, ranging from noisy training data to bad modelling choices, and different sources behave in different ways.  It turns out there are two fundamental types of uncertainty we need to take into account.
  # 
  # ### Aleatoric Uncertainty
  # 
  # Consider the following graph.  It shows the best fit linear regression to a set of ten data points.
  
  # In[2]:
  
  
  # Generate some fake data and plot a regression line.
  
  x = np.linspace(0, 5, 10)
  y = 0.15*x + np.random.random(10)
  plot.scatter(x, y)
  fit = np.polyfit(x, y, 1)
  line_x = np.linspace(-1, 6, 2)
  plot.plot(line_x, np.poly1d(fit)(line_x))
  plot.show()
  
  
  # The line clearly does not do a great job of fitting the data.  There are many possible reasons for this.  Perhaps the measuring device used to capture the data was not very accurate.  Perhaps `y` depends on some other factor in addition to `x`, and if we knew the value of that factor for each data point we could predict `y` more accurately.  Maybe the relationship between `x` and `y` simply isn't linear, and we need a more complicated model to capture it.  Regardless of the cause, the model clearly does a poor job of predicting the training data, and we need to keep that in mind.  We cannot expect it to be any more accurate on test data than on training data.  This is known as *aleatoric uncertainty*.
  # 
  # How can we estimate the size of this uncertainty?  By training a model to do it, of course!  At the same time it is learning to predict the outputs, it is also learning to predict how accurately each output matches the training data.  For every output of the model, we add a second output that produces the corresponding uncertainty.  Then we modify the loss function to make it learn both outputs at the same time.
  # 
  # ### Epistemic Uncertainty
  # 
  # Now consider these three curves.  They are fit to the same data points as before, but this time we are using 10th degree polynomials.
  
  # In[3]:
  
  
  plot.figure(figsize=(12, 3))
  line_x = np.linspace(0, 5, 50)
  for i in range(3):
      plot.subplot(1, 3, i+1)
      plot.scatter(x, y)
      fit = np.polyfit(np.concatenate([x, [3]]), np.concatenate([y, [i]]), 10)
      plot.plot(line_x, np.poly1d(fit)(line_x))
  plot.show()
  
  
  # Each of them perfectly interpolates the data points, yet they clearly are different models.  (In fact, there are infinitely many 10th degree polynomials that exactly interpolate any ten data points.)  They make identical predictions for the data we fit them to, but for any other value of `x` they produce different predictions.  This is called *epistemic uncertainty*.  It means the data does not fully constrain the model.  Given the training data, there are many different models we could have found, and those models make different predictions.
  # 
  # The ideal way to measure epistemic uncertainty is to train many different models, each time using a different random seed and possibly varying hyperparameters.  Then use all of them for each input and see how much the predictions vary.  This is very expensive to do, since it involves repeating the whole training process many times.  Fortunately, we can approximate the same effect in a less expensive way: by using dropout.
  # 
  # Recall that when you train a model with dropout, you are effectively training a huge ensemble of different models all at once.  Each training sample is evaluated with a different dropout mask, corresponding to a different random subset of the connections in the full model.  Usually we only perform dropout during training and use a single averaged mask for prediction.  But instead, let's use dropout for prediction too.  We can compute the output for lots of different dropout masks, then see how much the predictions vary.  This turns out to give a reasonable estimate of the epistemic uncertainty in the outputs.
  # 
  # ### Uncertain Uncertainty?
  # 
  # Now we can combine the two types of uncertainty to compute an overall estimate of the error in each output:
  # 
  # $$\sigma_\text{total} = \sqrt{\sigma_\text{aleatoric}^2 + \sigma_\text{epistemic}^2}$$
  # 
  # This is the value DeepChem reports.  But how much can you trust it?  Remember how I started this tutorial: deep learning models should not be used as black boxes.  We want to know how reliable the outputs are.  Adding uncertainty estimates does not completely eliminate the problem; it just adds a layer of indirection.  Now we have estimates of how reliable the outputs are, but no guarantees that those estimates are themselves reliable.
  # 
  # Let's go back to the example we started with.  We trained a model on the SAMPL training set, then generated predictions and uncertainties for the test set.  Since we know the correct outputs for all the test samples, we can evaluate how well we did.  Here is a plot of the absolute error in the predicted output versus the predicted uncertainty.
  
  # In[4]:
  
  
  abs_error = np.abs(y_pred.flatten()-test_dataset.y.flatten())
  plot.scatter(y_std.flatten(), abs_error)
  plot.xlabel('Standard Deviation')
  plot.ylabel('Absolute Error')
  plot.show()
  
  
  # The first thing we notice is that the axes have similar ranges.  The model clearly has learned the overall magnitude of errors in the predictions.  There also is clearly a correlation between the axes.  Values with larger uncertainties tend on average to have larger errors.
  # 
  # Now let's see how well the values satisfy the expected distribution.  If the standard deviations are correct, and if the errors are normally distributed (which is certainly not guaranteed to be true!), we expect 95% of the values to be within two standard deviations, and 99% to be within three standard deviations.  Here is a histogram of errors as measured in standard deviations.
  
  # In[5]:
  
  
  plot.hist(abs_error/y_std.flatten(), 20)
  plot.show()
  
  
  # Most of the values are in the expected range, but there are a handful of outliers at much larger values.  Perhaps this indicates the errors are not normally distributed, but it may also mean a few of the uncertainties are too low.  This is an important reminder: the uncertainties are just estimates, not rigorous measurements.  Most of them are pretty good, but you should not put too much confidence in any single value.
