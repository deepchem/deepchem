# DeepChem ADME

ADME (Absorption, Distribution, Metabolism, Excretion) is a core part of the drug discovery process. In-silico models\
 for ADME tasks span a wide variety of pharmacokinetics endpoints across multiple species.

The ADME benchmark contains three of the larger datasets that were released by AstraZenica on ChEMBL: human plasma pr\
otein binding (PPB), lipophilicity, and human clearance. While this data is small relative to full industrial dataset\
s, it is high quality and diverse.

Note that PPB dataset labels are transformed using %bound -> log(1 - %bound).

| Dataset | Examples | GC-DNN Val R2 (Scaffold Split) |
| ------ | ------ | ------ |
| Lipophilicty | 4200 | .653 |
| PPB | 1614 | .404 |
| Clearance | 1102 | .319 |

# Running Benchmark
```sh
$ python run_benchmark.py model split dataset
```

- models: {GraphConv, PDNN, RF, SVR}
- splits: {scaffold, random, index}
- dataset: {az_clearance.csv, az_hppb.csv, az_logd.csv}

Paper
----

www.arxiv.org/00000000

License
----

MIT