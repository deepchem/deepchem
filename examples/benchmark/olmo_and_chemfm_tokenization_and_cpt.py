import argparse
import gc
import os
import ssl
import time

os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_SHM_DISABLE", "1")

try:
    import certifi
    ssl._create_default_https_context = lambda: ssl.create_default_context(
        cafile=certifi.where())
except ImportError:
    pass

import deepchem as dc
from deepchem.models.torch_models.olmo import Olmo
from deepchem.models.lightning import LightningTorchModel
from lightning.pytorch.callbacks import EarlyStopping
from transformers import AutoTokenizer
import torch

from olmo_pretrain_benchmark import PRETRAINED_DIR as OLMO_CPT_DIR

DEFAULT_CHEMFM_TOKENIZER_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ChemFM", "finetuning",
    "property_prediction", "tokenizer")

if torch.cuda.is_available():
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))

def load_delaney():
    _, (train_dataset, _, test_dataset), _ = dc.molnet.load_delaney(
        featurizer=dc.feat.RawFeaturizer(smiles=True),
        splitter='scaffold',
        transformers=[])
    return train_dataset, test_dataset


def load_lipo():
    _, (train_dataset, _, test_dataset), _ = dc.molnet.load_lipo(
        featurizer=dc.feat.RawFeaturizer(smiles=True),
        splitter='scaffold',
        transformers=[])
    return train_dataset, test_dataset


def load_freesolv():
    _, (train_dataset, _, test_dataset), _ = dc.molnet.load_freesolv(
        featurizer=dc.feat.RawFeaturizer(smiles=True),
        splitter='scaffold',
        transformers=[])
    return train_dataset, test_dataset


def load_clearance():
    _, (train_dataset, _, test_dataset), _ = dc.molnet.load_clearance(
        featurizer=dc.feat.RawFeaturizer(smiles=True),
        splitter='scaffold',
        transformers=[])
    return train_dataset, test_dataset


def load_bace_pic50():
    _, (train_dataset, _, test_dataset), _ = dc.molnet.load_bace_regression(
        featurizer=dc.feat.RawFeaturizer(smiles=True),
        splitter='scaffold',
        transformers=[])
    return train_dataset, test_dataset

def load_sider():
    _, (train_dataset, _, test_dataset), _ = dc.molnet.load_sider(
        featurizer=dc.feat.RawFeaturizer(smiles=True),
        splitter='scaffold',
        transformers=[])
    return train_dataset, test_dataset


def load_clintox():
    _, (train_dataset, _, test_dataset), _ = dc.molnet.load_clintox(
        featurizer=dc.feat.RawFeaturizer(smiles=True),
        splitter='scaffold',
        transformers=[])
    return train_dataset, test_dataset

def load_bbbp():
    _, (train_dataset, _, test_dataset), _ = dc.molnet.load_bbbp(
        featurizer=dc.feat.RawFeaturizer(smiles=True),
        splitter='scaffold',
        transformers=[])
    return train_dataset, test_dataset


def load_bace():
    _, (train_dataset, _, test_dataset), _ = dc.molnet.load_bace_classification(
        featurizer=dc.feat.RawFeaturizer(smiles=True),
        splitter='scaffold',
        transformers=[])
    return train_dataset, test_dataset


def load_hiv():
    _, (train_dataset, _, test_dataset), _ = dc.molnet.load_hiv(
        featurizer=dc.feat.RawFeaturizer(smiles=True),
        splitter='scaffold',
        transformers=[])
    return train_dataset, test_dataset


def build_delaney_regression_dataset():
    train_dataset, test_dataset = load_delaney()
    return train_dataset, test_dataset, 1


def build_bace_regression_dataset():
    train_dataset, test_dataset = load_bace_pic50()
    return train_dataset, test_dataset, 1


def build_lipo_regression_dataset():
    train_dataset, test_dataset = load_lipo()
    return train_dataset, test_dataset, 1


def build_freesolv_regression_dataset():
    train_dataset, test_dataset = load_freesolv()
    return train_dataset, test_dataset, 1


def build_clearance_regression_dataset():
    train_dataset, test_dataset = load_clearance()
    return train_dataset, test_dataset, 1

def build_tox21_multitask_classification_dataset():
    tasks, (train_dataset, _, test_dataset), _ = dc.molnet.load_tox21(
        featurizer=dc.feat.RawFeaturizer(smiles=True),
        splitter='scaffold',
        transformers=[])
    return train_dataset, test_dataset, len(tasks)


def build_sider_multitask_classification_dataset():
    train_dataset, test_dataset = load_sider()
    return train_dataset, test_dataset, train_dataset.y.shape[1]


def build_clintox_multitask_classification_dataset():
    train_dataset, test_dataset = load_clintox()
    return train_dataset, test_dataset, train_dataset.y.shape[1]

def build_bbbp_classification_dataset():
    train_dataset, test_dataset = load_bbbp()
    return train_dataset, test_dataset, 1


def build_bace_classification_dataset():
    train_dataset, test_dataset = load_bace()
    return train_dataset, test_dataset, 1


def build_hiv_classification_dataset():
    train_dataset, test_dataset = load_hiv()
    return train_dataset, test_dataset, 1


REGRESSION_DATASETS = {
    "delaney": build_delaney_regression_dataset,
    "bace_regression": build_bace_regression_dataset,
    "lipo": build_lipo_regression_dataset,
    "freesolv": build_freesolv_regression_dataset,
    "clearance": build_clearance_regression_dataset,
}

MULTITASK_CLASSIFICATION_DATASETS = {
    "tox21": build_tox21_multitask_classification_dataset,
    "sider": build_sider_multitask_classification_dataset,
    "clintox": build_clintox_multitask_classification_dataset,
}


CLASSIFICATION_DATASETS = {
    "bbbp": build_bbbp_classification_dataset,
    "bace_classification": build_bace_classification_dataset,
    "hiv": build_hiv_classification_dataset,
}


OLMO_TASK_TYPES = {
    "regression": "regression",
    "multitask_classification": "mtc",
    "classification": "classification",
}

def modify_olmo_tokenizer_to_chemfm(model, chemfm_tokenizer_dir):
    chemfm_tokenizer = AutoTokenizer.from_pretrained(chemfm_tokenizer_dir)
    if chemfm_tokenizer.pad_token is None:
        chemfm_tokenizer.pad_token = chemfm_tokenizer.eos_token
    model.tokenizer = chemfm_tokenizer


def olmo_and_chemfm_tokenization_and_cpt(dataset_name="delaney",
                          nb_epoch=30,
                          batch_size=8,
                          pretrained_dir=OLMO_CPT_DIR,
                          chemfm_tokenizer_dir=DEFAULT_CHEMFM_TOKENIZER_DIR):
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    if dataset_name in REGRESSION_DATASETS:
        train_dataset, test_dataset, n_tasks = REGRESSION_DATASETS[
            dataset_name]()
        task_type = "regression"
    elif dataset_name in MULTITASK_CLASSIFICATION_DATASETS:
        train_dataset, test_dataset, n_tasks = MULTITASK_CLASSIFICATION_DATASETS[
            dataset_name]()
        task_type = "multitask_classification"
    elif dataset_name in CLASSIFICATION_DATASETS:
        train_dataset, test_dataset, n_tasks = CLASSIFICATION_DATASETS[
            dataset_name]()
        task_type = "classification"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"[{dataset_name}] Train size: {len(train_dataset)}, "
          f"Test size: {len(test_dataset)}, "
          f"Task type: {task_type}, "
          f"Number of tasks: {n_tasks}")

    model = Olmo(
        n_tasks=n_tasks,
        task_type=OLMO_TASK_TYPES[task_type],
        tokenizer_path=pretrained_dir,
        torch_dtype=dtype,
        batch_size=batch_size,
        finetune_strategy="qlora",
        gradient_checkpointing=True,
        skip_weight_init=True)

    modify_olmo_tokenizer_to_chemfm(model, chemfm_tokenizer_dir)
    model.load_from_pretrained(pretrained_dir, from_hf_checkpoint=True)

    early_stopping_callback = EarlyStopping(
        monitor="train_loss", patience=3, mode="min")

    num_gpus = torch.cuda.device_count()
    lightning_model = LightningTorchModel(
        model=model,
        batch_size=batch_size,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=-1 if torch.cuda.is_available() else 1,
        strategy="ddp" if num_gpus > 1 else "auto",
        callbacks=[early_stopping_callback])

    lightning_model.fit(train_dataset, nb_epoch=nb_epoch, num_workers=0)
    model.model.to(model.device)

    metric_name = "rms_score" if task_type == "regression" else "roc_auc_score"
    metric = dc.metrics.Metric(dc.metrics.rms_score
                               if task_type == "regression" else dc.metrics.roc_auc_score)

    test_score = model.evaluate(test_dataset, metrics=[metric])[metric_name]
    print(f"[{dataset_name}] Test {metric_name}: {test_score:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="delaney",
                        choices=list(REGRESSION_DATASETS.keys()) +
                        list(MULTITASK_CLASSIFICATION_DATASETS.keys()) +
                        list(CLASSIFICATION_DATASETS.keys()),
                        help="Dataset name")
    parser.add_argument("--nb_epoch", type=int, default=30,
                        help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--pretrained_dir", type=str,
                        default=OLMO_CPT_DIR,
                        help="HuggingFace model ID or local directory to "
                        "load the OLMo checkpoint and tokenizer from. "
                        "Defaults to the continued-pretraining backbone "
                        "produced by olmo_pretrain_benchmark.py "
                        f"({OLMO_CPT_DIR}); run that script first to "
                        "produce it.")
    parser.add_argument("--chemfm_tokenizer_dir", type=str,
                        default=DEFAULT_CHEMFM_TOKENIZER_DIR,
                        help="HuggingFace model ID or local directory "
                        "containing ChemFM's tokenizer files "
                        "(tokenizer.json, tokenizer_config.json)")
    args = parser.parse_args()

    olmo_and_chemfm_tokenization_and_cpt(dataset_name=args.dataset,
                              nb_epoch=args.nb_epoch,
                              batch_size=args.batch_size,
                              pretrained_dir=args.pretrained_dir,
                              chemfm_tokenizer_dir=args.chemfm_tokenizer_dir)