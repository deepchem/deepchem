import deepchem as dc
import tensorflow as tf
import numpy as np
import os
import json
import time


def input_fn(dataset, epochs):
    x, y, weights = dataset.make_iterator(batch_size=100, epochs=epochs).get_next()
    return {'x': x, 'weights': weights}, y

def mean_auc(labels, predictions, weights):
    metric_ops = []
    update_ops = []
    for i in range(n_tasks):
        metric, update = tf.metrics.auc(labels[:,i], predictions[:,i], weights[:,i])
        metric_ops.append(metric)
        update_ops.append(update)

    mean_metric = tf.reduce_mean(tf.stack(metric_ops))
    update_all = tf.group(*update_ops)
    return mean_metric, update_all

def run():
   os.environ['GRPC_POLL_STRATEGY'] = 'poll'
   tf.logging.set_verbosity(tf.logging.DEBUG)
   try:
       task_type = os.environ['JOB_NAME']
       task_index = int(os.environ['TASK_INDEX'])
       ps_hosts = os.environ['PS_HOSTS'].split(',')
       worker_hosts = os.environ['WORKER_HOSTS'].split(',')
       TF_CONFIG = {
          'task': {'type': task_type, 'index': task_index},
          'cluster': {
              'chief': [worker_hosts[0]],
              'worker': worker_hosts,
              'ps': ps_hosts
          },
          'environment': 'cloud'
       }

       local_ip = 'localhost:' + TF_CONFIG['cluster'][task_type][task_index].split(':')[1]
       TF_CONFIG['cluster'][task_type][task_index] = local_ip
       if (task_type in ('chief', 'master')) or (task_type == 'worker' and task_index == 0):
           TF_CONFIG['cluster']['worker'][task_index] = local_ip
           TF_CONFIG['task']['type'] = 'chief'

       os.environ['TF_CONFIG'] = json.dumps(TF_CONFIG)
   except KeyError as ex:
       print(ex)
       job_name = None
       task_index = 0
       ps_hosts = None
       worker_hosts = None

   tasks, datasets, transformers = dc.molnet.load_tox21()
   train_dataset, valid_dataset, test_dataset = datasets
   n_tasks = len(tasks)
   n_features = train_dataset.X.shape[1]

   model = dc.models.MultitaskClassifier(n_tasks, n_features, layer_sizes=[1000], dropout=0.25)

   print "featurizing columns"
   x_col = tf.feature_column.numeric_column('x', shape=(n_features,))
   weight_col = tf.feature_column.numeric_column('weights', shape=(n_tasks,))

   print "entering estimator"
   estimator = model.make_estimator(feature_columns=[x_col], weight_column=weight_col, metrics={'mean_auc': mean_auc},
                                 model_dir='/logs')

   # following lines added to run train_and_evaluate function of deepchem which is compatible for distributed training
   train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(train_dataset, 100), max_steps=100000)
   eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(test_dataset, 1), steps=None, start_delay_secs=0,
                                  throttle_secs=30)

   tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

if __name__ == "__main__":
  run()
