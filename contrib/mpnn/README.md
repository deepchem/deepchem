# Message Passing Neural Networks

MPNNs aim to generalize molecular machine learning models that operate on graph-valued inputs. Graph-Convolutions [https://arxiv.org/abs/1509.09292] and Weaves \
[https://arxiv.org/abs/1603.00856] (among others) can be recast into this framework [https://arxiv.org/abs/1704.01212]

The premise is that the featurization of arbitrary chemical multigraphs can be broken down into a message function, vertex-update function, and a readout functi\
on that is invariant to graph isomorphisms. All functions must be subdifferentiable to preserve gradient-flow and ideally are learnable as well

Models of this style introduce an additional parameter **T**, which is the number of iterations for the message-passing stage. Values greater than 4 don't seem \
to improve performance.

##MPNN-S Variant
 MPNNs do provide a nice mathematical framework that can capture modern molecular machine learning algorithms we work with today. One criticism of this algorithm class is that training is slow, due to the sheer number of training iterations required for convergence - at batch size 20 on QM9, the MPNN authors trained for 540 epochs.
 
This can be improved significantly by using batch normalization, or more interestingly, the new SELU activation [https://arxiv.org/pdf/1706.02515.pdf]. In order to use SELUs straight through the system, we dropped the GRU unit [https://arxiv.org/pdf/1412.3555.pdf] the authors used in favor of a SELU activated fully-connected neural network for each time step **T**. This modified approach now achieves peak performance in as little as 60 epochs on most molecular machine learning datasets.

MPNN-S sets new records on the Delaney & PPB datasets:

| Dataset | Num Examples | MP-DNN Val R2 [Scaffold Split] | GraphConv Val R2 [Scaffold Split] |
| ------ | ------ | ------ | ------ |
| Delaney | 1102 | **.820** | .606 |
| PPB | 1600 | **.427** | .381 |
| Clearance | 838 | **.32** | .28 |


## Run Code
```sh
$ python mpnn.py
```

License
----

MIT
