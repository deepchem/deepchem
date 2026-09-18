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
import torch

if torch.cuda.is_available():
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))

PRETRAINED_DIR = "./olmo_pretrained_backbone"
CLASSIFICATION_FINETUNE_DIR = "./olmo_checkpoints_classification"


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


def build_bbbp_classification_dataset():
    train_dataset, test_dataset = load_bbbp()
    return train_dataset, test_dataset, 1


def build_bace_classification_dataset():
    train_dataset, test_dataset = load_bace()
    return train_dataset, test_dataset, 1


def build_hiv_classification_dataset():
    train_dataset, test_dataset = load_hiv()
    return train_dataset, test_dataset, 1


CLASSIFICATION_DATASETS = {
    "bbbp": build_bbbp_classification_dataset,
    "bace": build_bace_classification_dataset,
    "hiv": build_hiv_classification_dataset,
}


def finetune_classification(dataset_name="bbbp",
                            nb_epoch=5,
                            batch_size=8,
                            pretrained_dir=PRETRAINED_DIR):
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    train_dataset, test_dataset, n_tasks = CLASSIFICATION_DATASETS[
        dataset_name]()
    print(f"[{dataset_name}] Train size: {len(train_dataset)}, "
          f"Test size: {len(test_dataset)}, n_tasks: {n_tasks}")

    model_dir = f"{CLASSIFICATION_FINETUNE_DIR}_{dataset_name}"
    finetune_model = Olmo(task_type="classification",
                          tokenizer_path="allenai/OLMo-1B-hf",
                          n_tasks=n_tasks,
                          torch_dtype=dtype,
                          finetune_strategy="qlora",
                          gradient_checkpointing=True,
                          model_dir=model_dir,
                          batch_size=batch_size,
                          learning_rate=2e-4,
                          skip_weight_init=True)

    print(f"Loading pretrained backbone from {pretrained_dir}")
    finetune_model.load_from_pretrained(pretrained_dir, from_hf_checkpoint=True)

    metric = dc.metrics.Metric(dc.metrics.roc_auc_score)

    baseline_auc = finetune_model.evaluate(test_dataset,
                                           metrics=[metric])["roc_auc_score"]
    print(
        f"Baseline test ROC-AUC (random classification head, pretrained backbone): "
        f"{baseline_auc:.3f}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    num_gpus = torch.cuda.device_count()
    early_stopping = EarlyStopping(monitor="train_loss",
                                   mode="min",
                                   patience=7)
    trainer = LightningTorchModel(
        model=finetune_model,
        batch_size=batch_size,
        model_dir=model_dir,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=-1 if torch.cuda.is_available() else 1,
        strategy="ddp" if num_gpus > 1 else "auto",
        enable_progress_bar=True,
        log_every_n_steps=1,
        callbacks=[early_stopping])

    t0 = time.time()
    trainer.fit(train_dataset, nb_epoch=nb_epoch, num_workers=0)
    elapsed = time.time() - t0

    finetune_model.model.to(finetune_model.device)

    train_auc = finetune_model.evaluate(train_dataset,
                                        metrics=[metric])["roc_auc_score"]
    test_auc = finetune_model.evaluate(test_dataset,
                                       metrics=[metric])["roc_auc_score"]
    peak_mem = (torch.cuda.max_memory_allocated() /
                1e9 if torch.cuda.is_available() else 0.0)

    print(
        f"[{dataset_name}] After {nb_epoch} epochs on {num_gpus} GPU(s) "
        f"({elapsed:.1f}s): train_auc={train_auc:.3f} test_auc={test_auc:.3f} "
        f"peak_mem={peak_mem:.2f}GB")
    print(f"Checkpoints saved under {os.path.join(model_dir, 'checkpoints')}")

    del finetune_model, trainer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        choices=list(CLASSIFICATION_DATASETS),
                        default=None,
                        help="Run a single dataset instead of all of them.")
    parser.add_argument(
        "--pretrained-dir",
        default=PRETRAINED_DIR,
        help="Backbone to finetune from, produced by "
        "`olmo_pretrain_benchmark.py` (continued pretraining on streamed "
        f"UniChem molecules). Defaults to {PRETRAINED_DIR}.")
    args = parser.parse_args()

    if args.dataset:
        finetune_classification(args.dataset,
                                pretrained_dir=args.pretrained_dir)
    else:
        for dataset_name in CLASSIFICATION_DATASETS:
            finetune_classification(dataset_name,
                                    pretrained_dir=args.pretrained_dir)
