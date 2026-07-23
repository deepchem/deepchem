"""
Contains class for hyperparameter optimization using the hyperopt library.
"""
import os
import tempfile
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable

import numpy as np

from deepchem.data import Dataset
from deepchem.trans import Transformer
from deepchem.models import Model
from deepchem.metrics import Metric
from deepchem.hyper.base_classes import HyperparamOpt
from deepchem.hyper.base_classes import _convert_hyperparam_dict_to_filename

logger = logging.getLogger(__name__)


class HyperoptHyperparamOpt(HyperparamOpt):
    """Provides hyperparameter search using the hyperopt library.

    This class uses the Tree-structured Parzen Estimator (TPE) algorithm from
    the `hyperopt <https://github.com/hyperopt/hyperopt>`_ library to perform a
    guided (Bayesian) search over the hyperparameter space. Unlike
    `GridHyperparamOpt` (which searches exhaustively) or `RandomHyperparamOpt`
    (which samples independently), this optimizer uses the results of previous
    evaluations to decide which hyperparameters to try next, which is often much
    more sample efficient for expensive-to-train models.

    The search space is specified using hyperopt's expression objects (for
    example `hyperopt.hp.uniform` for continuous ranges and `hyperopt.hp.choice`
    for a discrete set of options). See the hyperopt documentation for the full
    list of available expressions.

    Note that this class requires hyperopt to be installed.

    Examples
    --------
    This example shows the type of constructor function this class expects.

    >>> import deepchem as dc
    >>> import numpy as np
    >>> from sklearn.ensemble import RandomForestRegressor as RF
    >>> from hyperopt import hp
    >>> # generating data
    >>> X = np.random.rand(50, 5)
    >>> y = np.random.rand(50, 1)
    >>> dataset = dc.data.NumpyDataset(X, y)
    >>> # splitting dataset into train and validation
    >>> splitter = dc.splits.RandomSplitter()
    >>> train_dataset, valid_dataset = splitter.train_test_split(dataset)
    >>> # metric to evaluate a set of parameters
    >>> metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)
    >>> # defining `model_builder`
    >>> def model_builder(**model_params):
    ...   n_estimators = model_params['n_estimators']
    ...   model_dir = model_params['model_dir']
    ...   rf = RF(n_estimators=n_estimators)
    ...   return dc.models.SklearnModel(rf, model_dir)
    >>> # the search space, specified using hyperopt expressions
    >>> params_dict = {'n_estimators': hp.choice('n_estimators', [10, 100])}
    >>> # creating optimizer and searching over hyperparameters
    >>> optimizer = dc.hyper.HyperoptHyperparamOpt(model_builder, max_evals=5)
    >>> best_model, best_hyperparams, all_results = optimizer.hyperparam_search(
    ...     params_dict, train_dataset, valid_dataset, metric)  # doctest: +SKIP

    Parameters
    ----------
    model_builder: constructor function.
        This parameter must be constructor function which returns an
        object which is an instance of `dc.models.Model`. This function
        must accept two arguments, `model_params` of type `dict` and
        `model_dir`, a string specifying a path to a model directory.
    max_evals: int, (default 10)
        The maximum number of parameter settings (model trainings) that
        hyperopt is allowed to evaluate during the search.
    """

    def __init__(self,
                 model_builder: Callable[..., Model],
                 max_evals: int = 10):
        super(HyperoptHyperparamOpt, self).__init__(model_builder=model_builder)
        self.max_evals = max_evals

    def hyperparam_search(
        self,
        params_dict: Dict,
        train_dataset: Dataset,
        valid_dataset: Dataset,
        metric: Metric,
        output_transformers: List[Transformer] = [],
        nb_epoch: int = 10,
        use_max: bool = True,
        logfile: str = 'results.txt',
        logdir: Optional[str] = None,
        **kwargs,
    ) -> Tuple[Model, Dict[str, Any], Dict[str, Any]]:
        """Perform hyperparameter search using hyperopt.

        Parameters
        ----------
        params_dict: Dict
            Maps each hyperparameter name (string) to a hyperopt search
            expression describing the values to search over, for example
            ``{'learning_rate': hp.uniform('learning_rate', 1e-4, 1e-1)}``.
        train_dataset: Dataset
            dataset used for training
        valid_dataset: Dataset
            dataset used for validation (optimization on valid scores)
        metric: Metric
            metric used for evaluation
        output_transformers: list[Transformer]
            Transformers for evaluation. This argument is needed since
            `train_dataset` and `valid_dataset` may have been transformed
            for learning and need the transform to be inverted before
            the metric can be evaluated on a model.
        nb_epoch: int, (default 10)
            Specifies the number of training epochs during each iteration of
            optimization. Not used by all model types.
        use_max: bool, optional
            If True, return the model with the highest score. Else return
            model with the minimum score. Since hyperopt always minimizes its
            objective, the validation score is negated internally when
            `use_max` is True.
        logdir: str, optional
            The directory in which to store created models. If not set, will
            use a temporary directory.
        logfile: str, optional (default `results.txt`)
            Name of logfile to write results to. If specified, this must
            be a valid file name. If not specified, results of hyperparameter
            search will be written to `logdir/results.txt`.

        Returns
        -------
        Tuple[`best_model`, `best_hyperparams`, `all_scores`]
            `(best_model, best_hyperparams, all_scores)` where `best_model` is
            an instance of `dc.models.Model`, `best_hyperparams` is a
            dictionary of parameters, and `all_scores` is a dictionary mapping
            string representations of hyperparameter sets to validation
            scores.
        """
        try:
            from hyperopt import fmin, tpe, Trials, STATUS_OK
        except ModuleNotFoundError:
            raise ImportError("This class requires hyperopt to be installed.")

        if logdir is not None:
            if not os.path.exists(logdir):
                os.makedirs(logdir, exist_ok=True)
            log_file = os.path.join(logdir, logfile)

        all_scores: Dict[str, Any] = {}
        # Track the best result across trials. hyperopt's `fmin` only returns the
        # best hyperparameters (and, for hp.choice, as indices rather than
        # values), so we keep track of the actual best model ourselves.
        best: Dict[str, Any] = {
            'score': -np.inf if use_max else np.inf,
            'model': None,
            'hyperparams': None,
        }

        def objective(model_params: Dict) -> Dict[str, Any]:
            model_params = dict(model_params)
            hp_str = _convert_hyperparam_dict_to_filename(model_params)
            logger.info("Fitting model with hyperparameters: %s" %
                        str(model_params))

            if logdir is not None:
                model_dir = os.path.join(logdir, hp_str)
                try:
                    os.makedirs(model_dir)
                except OSError:
                    if not os.path.isdir(model_dir):
                        logger.info(
                            "Error creating model_dir, using tempfile directory"
                        )
                        model_dir = tempfile.mkdtemp()
            else:
                model_dir = tempfile.mkdtemp()

            model_params['model_dir'] = model_dir
            model = self.model_builder(**model_params)

            # mypy test throws error, so ignoring it in try
            try:
                model.fit(train_dataset, nb_epoch=nb_epoch)  # type: ignore
            # Not all models have nb_epoch
            except TypeError:
                model.fit(train_dataset)
            try:
                model.save()
            # Some models autosave
            except NotImplementedError:
                pass

            multitask_scores = model.evaluate(valid_dataset, [metric],
                                              output_transformers)
            valid_score = multitask_scores[metric.name]
            all_scores[hp_str] = valid_score

            if (use_max and valid_score >= best['score']) or (
                    not use_max and valid_score <= best['score']):
                best['score'] = valid_score
                best['model'] = model
                best['hyperparams'] = model_params

            logger.info("Model %s, Validation set score: %f" %
                        (hp_str, valid_score))
            logger.info("\tbest_validation_score so far: %f" % best['score'])

            # hyperopt always minimizes, so negate the score when maximizing.
            loss = -valid_score if use_max else valid_score
            return {'loss': loss, 'status': STATUS_OK}

        trials = Trials()
        fmin(objective,
             space=params_dict,
             algo=tpe.suggest,
             max_evals=self.max_evals,
             trials=trials)

        best_model = best['model']
        best_hyperparams = best['hyperparams']

        if best_model is None:
            logger.info("No models trained correctly.")
            if logdir is not None:
                with open(log_file, 'w+') as f:
                    f.write(
                        "No model trained correctly. Arbitrary models returned")
            # mypy: best_model is narrowed to None here, but the declared return
            # type is Model; this defensive branch mirrors the other optimizers.
            return best_model, best_hyperparams, all_scores  # type: ignore

        multitask_scores = best_model.evaluate(train_dataset, [metric],
                                               output_transformers)
        train_score = multitask_scores[metric.name]
        logger.info("Best hyperparameters: %s" % str(best_hyperparams))
        logger.info("best train_score: %f" % train_score)
        logger.info("best validation_score: %f" % best['score'])

        if logdir is not None:
            with open(log_file, 'w+') as f:
                f.write("Best Hyperparameters dictionary %s\n" %
                        str(best_hyperparams))
                f.write("Best validation score %f\n" % best['score'])
                f.write("Best train_score: %f\n" % train_score)
        return best_model, best_hyperparams, all_scores
