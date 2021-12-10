# flake8: noqa

# metric class
from deepchem.metrics.metric import Metric
# metrics utils
from deepchem.metrics.metric import threshold_predictions
from deepchem.metrics.metric import normalize_weight_shape
from deepchem.metrics.metric import normalize_labels_shape
from deepchem.metrics.metric import normalize_prediction_shape
from deepchem.metrics.metric import handle_classification_mode
from deepchem.metrics.metric import to_one_hot
from deepchem.metrics.metric import from_one_hot

# sklearn & scipy score function
from deepchem.metrics.score_function import matthews_corrcoef
from deepchem.metrics.score_function import recall_score
from deepchem.metrics.score_function import kappa_score
from deepchem.metrics.score_function import cohen_kappa_score
from deepchem.metrics.score_function import r2_score
from deepchem.metrics.score_function import mean_squared_error
from deepchem.metrics.score_function import mean_absolute_error
from deepchem.metrics.score_function import precision_score
from deepchem.metrics.score_function import precision_recall_curve
from deepchem.metrics.score_function import auc
from deepchem.metrics.score_function import jaccard_score
from deepchem.metrics.score_function import f1_score
from deepchem.metrics.score_function import roc_auc_score
from deepchem.metrics.score_function import accuracy_score
from deepchem.metrics.score_function import balanced_accuracy_score
from deepchem.metrics.score_function import top_k_accuracy_score
from deepchem.metrics.score_function import pearsonr

# original score function
from deepchem.metrics.score_function import pearson_r2_score
from deepchem.metrics.score_function import jaccard_index
from deepchem.metrics.score_function import pixel_error
from deepchem.metrics.score_function import prc_auc_score
from deepchem.metrics.score_function import rms_score
from deepchem.metrics.score_function import mae_score
from deepchem.metrics.score_function import bedroc_score
from deepchem.metrics.score_function import concordance_index
