# Message Passing Neural Networks

MPNNs aim to generalize molecular machine learning models that operate on graph-valued inputs. Graph-Convolutions [https://arxiv.org/abs/1509.09292] and Weaves [https://arxiv.org/abs/1603.00856] (among others) can be recast into this framework [https://arxiv.org/abs/1704.01212]

The premise is that the featurization of arbitrary chemical multigraphs can be broken down into a message function, vertex-update function, and a readout function that is invariant to graph isomorphisms. All functions must be subdifferentiable to preserve gradient-flow and ideally are learnable too.

Models of this style introduce an additional parameter **T**, which is the number of iterations for the message-passing stage. Values greater than 4 don't seem to improve performance.

Requires PyTorch.

| Dataset | Examples | MP-DNN Val R2 (Index Split) |
| ------ | ------ | ------ |
| Delaney | 1102 | .801 |

## Running Code
```sh
$ python mpnn_baseline.py
```

License
----

MIT