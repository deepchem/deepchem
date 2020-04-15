# Low Data Examples

This example shows how to apply the MAML (Model-Agnostic
Meta-Learning) algorithm to chemical datasets. The original MAML algorithm was introduced in the following paper

Finn, Chelsea, Pieter Abbeel, and Sergey Levine. "Model-agnostic meta-learning for fast adaptation of deep networks." Proceedings of the 34th International Conference on Machine Learning-Volume 70. JMLR. org, 2017.

We use the `dc.metalearning.MAML` implementation for our
example. The basic setup we have is that we have training data
for a variety of tasks, but only a small amount for each one.
MAML performs meta-learning by looping over tasks, and trying to
minimize the loss on each one *after* one or a few steps of
gradient descent. That is, it doesn't try to create a model
which directly solves the tasks but rather a model that is very
easy to train.

To use this class, create a subclass of `MetaLearner` that
encapsulates the model and data for your learning problem.
Pass it to a MAML object and call `fit()`.  You can then use
`train_on_current_task()` to fine tune the model for a
particular task.

In this example, we train on the ToxCast dataset from
MoleculeNet that has a large number of tasks. We create a
subclass `ToxcastLearner` of `MetaLearner` that we use to do our
learning.

Toxcast has data on 6874 molecules and 617 tasks.  However, the
data is very sparse: most tasks do not include data for most
molecules.  It also is very unbalanced: there are many more
negatives than positives.  For each task, the example creates a
list of alternating positives and negatives so each batch will
have equal numbers of both.

Here's results from a trial run:

Before fine tuning:

Cross entropy loss: 0.6707157
Prediction accuracy: 0.6443548387096775
ROC AUC: 0.7065296566077003

After fine tuning:

Cross entropy loss: 0.6586808
Prediction accuracy: 0.6653225806451613
ROC AUC: 0.7135145681581685

There's a models bump we see from the fine tuning for MAML.
