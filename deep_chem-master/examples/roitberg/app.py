import numpy as np
from flask import request, abort, Flask

import flask

webapp = Flask(__name__)


@webapp.route('/potential', methods=["POST"])
def potential():
  content = request.get_json(force=True)
  if not content or not 'X' in content:
    abort(400)
  X = np.array(content['X'])
  x0 = X[:, 1:]
  a0 = X[:, :1]
  result = webapp.model.pred_one(x0, a0)
  return flask.jsonify({'y': result.tolist()[0]}), 200


@webapp.route('/gradient', methods=["POST"])
def index():
  content = request.get_json(force=True)
  if not content or not 'X' in content:
    abort(400)
  X = np.array(content['X'])
  num_atoms = X.shape[0]
  x0 = X[:, 1:]
  a0 = X[:, :1]

  res = webapp.model.grad_one(x0, a0)
  res = res.reshape((num_atoms, 3))

  return flask.jsonify({'grad': res.tolist()}), 200


@webapp.route('/minimize', methods=["POST"])
def minimize():
  content = request.get_json(force=True)
  if not content or not 'X' in content:
    abort(400)
  X = np.array(content['X'])

  constraints = None

  if 'constraints' in content:
    constraints = content['constraints']
    print('setting constraints')

  num_atoms = X.shape[0]
  x0 = X[:, 1:]
  a0 = X[:, :1]

  res = webapp.model.minimize_structure(x0, a0, constraints)
  res = res.reshape((num_atoms, 3))
  y = webapp.model.pred_one(res, a0).tolist()[0]

  return flask.jsonify({'X': res.tolist(), 'y': y}), 200
