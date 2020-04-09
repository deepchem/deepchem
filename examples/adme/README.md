# ADME Dataset Examples

ADME (Absorption, Distribution, Metabolism, Excretion) is a core
part of the drug discovery process. In-silico models for ADME
tasks span a wide variety of pharmacokinetics endpoints across
multiple species.

The ADME benchmark contains three of the larger datasets that
were released by AstraZeneca on ChEMBL: human plasma protein
binding (PPB), lipophilicity, and human clearance. While this
data is small relative to full industrial datasets, it is high
quality and diverse.

Note that PPB dataset labels are transformed using %bound -> log(1 - %bound).

| Dataset | Examples | GC-DNN Val R2 (Scaffold Split) |
| ------ | ------ | ------ |
| Lipophilicty | 4200 | .653 |
| PPB | 1614 | .404 |
| Clearance | 1102 | .319 |

# Running Benchmark
```sh
$ python run_benchmark.py
```

You can manually edit variables within `run_benchmarks.py` to the following values and fun

- `MODEL`: {GraphConv, RF, SVR}
- `SPLIT`: {scaffold, random, index}
- `DATASET_NAME`: {clearance, hppb}

License
----

MIT
