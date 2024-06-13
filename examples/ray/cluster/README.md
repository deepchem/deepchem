# Using Ray Cluster for multi-node featurization

Steps:
1. Create `working_dir` and populate it with data and training script.
2. Execute the following commands:
``py
# spin up cluster
ray up -y config.yaml
# connect to dashboard in local machine (though this step can be skipped)
ray dashboard config.yaml
# submit job
ray job submit --address http://localhost:8265 --working-dir working_dir -- python ray_dataset.py
# spin down cluster
ray down config.yaml
``

To view ray dashboard: `ray dashboard config.yaml`

To connect to the instance (via ssh) on the terminal: `ray attach config.yaml`
