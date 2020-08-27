"""
Scikit-learn wrapper interface of xgboost
"""

import os
import logging
import tempfile
from typing import Any, Dict, Optional, Union

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV

from deepchem.data import Dataset
from deepchem.models.sklearn_models import SklearnModel

logger = logging.getLogger(__name__)


class XGBoostModel(SklearnModel):
  """
  Scikit-learn wrapper class for XGBoost model.

  Notes
  -----
  This class require XGBoost to be installed.
  """

  def __init__(self,
               model_instance: Union[xgb.XGBClassifier, xgb.XGBRegressor],
               model_dir: Optional[str] = None,
               **kwargs):
    """
    Parameters
    ----------
    model_instance: Union[xgb.XGBClassifier, xgb.XGBRegressor]
      Scikit-learn wrapper interface of XGBoost models.
    model_dir: str, optional (default None)
      Path to directory where model will be stored.
    """
    if model_dir is not None:
      if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    else:
      model_dir = tempfile.mkdtemp()
    self.model_dir = model_dir
    self.model_instance = model_instance
    self.model_class = model_instance.__class__

    if 'early_stopping_rounds' in kwargs:
      self.early_stopping_rounds = kwargs['early_stopping_rounds']
    else:
      self.early_stopping_rounds = 50

  # FIXME: Return type "None" of "fit" incompatible with return type "float" in supertype "Model"
  def fit(self, dataset: Dataset, **kwargs) -> None:  # type: ignore[override]
    """Fits XGBoost model to data.

    dataset: Dataset
      The `Dataset` to train this model on.
    """
    X = dataset.X
    y = np.squeeze(dataset.y)
    seed = self.model_instance.random_state
    if isinstance(self.model_instance, xgb.XGBClassifier):
      xgb_metric = "auc"
      sklearn_metric = "roc_auc"
      stratify = y
    elif isinstance(self.model_instance, xgb.XGBRegressor):
      xgb_metric = "mae"
      sklearn_metric = "neg_mean_absolute_error"
      stratify = None
    best_param = self._search_param(sklearn_metric, X, y)
    # update model with best param
    self.model_instance = self.model_class(**best_param)

    # Find optimal n_estimators based on original learning_rate
    # and early_stopping_rounds
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=stratify)

    self.model_instance.fit(
        X_train,
        y_train,
        early_stopping_rounds=self.early_stopping_rounds,
        eval_metric=xgb_metric,
        eval_set=[(X_train, y_train), (X_test, y_test)])

    # Since test size is 20%, when retrain model to whole data, expect
    # n_estimator increased to 1/0.8 = 1.25 time.
    estimated_best_round = np.round(self.model_instance.best_ntree_limit * 1.25)
    self.model_instance.n_estimators = np.int64(estimated_best_round)
    self.model_instance.fit(X, y, eval_metric=xgb_metric)

  def _search_param(self, metric: str, X: np.ndarray,
                    y: np.ndarray) -> Dict[str, Any]:
    """Find best potential parameters set using few n_estimators"""

    # Make sure user specified params are in the grid.

    def unique_not_none(values):
      return list(np.unique([x for x in values if x is not None]))

    max_depth_grid = unique_not_none([self.model_instance.max_depth, 5, 7])
    colsample_bytree_grid = unique_not_none(
        [self.model_instance.colsample_bytree, 0.66, 0.9])
    reg_lambda_grid = unique_not_none([self.model_instance.reg_lambda, 1, 5])
    learning_rate = 0.3
    if self.model_instance.learning_rate is not None:
      learning_rate = max(learning_rate, self.model_instance.learning_rate)
    n_estimators = 60
    if self.model_instance.n_estimators is not None:
      n_estimators = min(n_estimators, self.model_instance.n_estimators)
    param_grid = {
        'max_depth': max_depth_grid,
        'learning_rate': [learning_rate],
        'n_estimators': [n_estimators],
        'gamma': [self.model_instance.gamma],
        'min_child_weight': [self.model_instance.min_child_weight],
        'max_delta_step': [self.model_instance.max_delta_step],
        'subsample': [self.model_instance.subsample],
        'colsample_bytree': colsample_bytree_grid,
        'colsample_bylevel': [self.model_instance.colsample_bylevel],
        'reg_alpha': [self.model_instance.reg_alpha],
        'reg_lambda': reg_lambda_grid,
        'scale_pos_weight': [self.model_instance.scale_pos_weight],
        'base_score': [self.model_instance.base_score],
        'seed': [self.model_instance.random_state]
    }
    grid_search = GridSearchCV(
        self.model_instance, param_grid, cv=2, refit=False, scoring=metric)
    grid_search.fit(X, y)
    best_params = grid_search.best_params_
    # Change params back original params
    best_params['learning_rate'] = self.model_instance.learning_rate
    best_params['n_estimators'] = self.model_instance.n_estimators
    return best_params
