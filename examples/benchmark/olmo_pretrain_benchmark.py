import argparse
import gc
import os
import shutil
import ssl

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
from datasets import load_dataset
from rdkit import Chem
import numpy as np
import torch

if torch.cuda.is_available():
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))

MAX_SAMPLES = 5000  # subset for continued pretraining
MAX_SMILES_LENGTH = 200  # cap on longer SMILES strings to avoid OOM
PRETRAINED_DIR = "./olmo_pretrained_backbone"
CHECKPOINT_DIR = "./olmo_checkpoints_causal_lm"


def canonicalize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def build_pretraining_safe_gpt_dataset():
    dataset = load_dataset("datamol-io/safe-gpt",
                           split="train",
                           streaming=True)

    smiles = []
    seen = set()

    print(f"Streaming UniChem...")
    print(f"Target: {MAX_SAMPLES:,} unique molecules")

    for row in dataset:
        if str(row.get("source", "")).lower() != "unichem":
            continue

        molecule = row.get("smiles")
        if not molecule:
            continue

        molecule = canonicalize_smiles(molecule)
        if molecule is None:
            continue

        if len(molecule) > MAX_SMILES_LENGTH:
            continue

        if molecule in seen:
            continue

        seen.add(molecule)
        smiles.append(molecule)

        if len(smiles) % 10_000 == 0:
            print(f"Collected {len(smiles):,}/{MAX_SAMPLES:,} molecules")

        if len(smiles) >= MAX_SAMPLES:
            break

    if len(smiles) == 0:
        raise RuntimeError(
            "No UniChem molecules were found in datamol-io/safe-gpt.")

    smiles = np.asarray(smiles, dtype=object)

    print(f"Final pretraining corpus: {len(smiles):,} molecules")

    return dc.data.DiskDataset.from_numpy(X=smiles, y=smiles)


def run_generation(hf_model, quantized):
    prompts = [
        "OCC3OC(OCC2OC(OC(C#N)c1ccccc1)C(O)C(O)C2O)C(O)C(O)C3O.",
        "Cc1occc1C(=O)Nc2ccccc2."
    ]

    def _restore_from_lightning_checkpoint():
        LightningTorchModel(model=hf_model,
                            batch_size=hf_model.batch_size,
                            model_dir=hf_model.model_dir).restore(strict=False)

    if quantized:
        _restore_from_lightning_checkpoint()
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        hf_model.model.to("cpu")
        torch.cuda.empty_cache()
        _restore_from_lightning_checkpoint()
        hf_model.model.to(device)

    # Sanity check for NaN or Inf in the logits before generation

    hf_model.model.eval()
    with torch.no_grad():
        probe = hf_model.tokenizer(prompts, padding=True, return_tensors="pt")
        probe = {k: v.to(hf_model.device) for k, v in probe.items()}
        logits = hf_model.model(**probe).logits
        print(f"[diagnostic] forward logits: "
              f"any_nan={torch.isnan(logits).any().item()} "
              f"any_inf={torch.isinf(logits).any().item()} "
              f"min={logits.min().item():.3f} max={logits.max().item():.3f}")

    prompt_dataset = dc.data.NumpyDataset(X=np.array(prompts))
    outputs = hf_model.generate(prompt_dataset,
                                max_length=100,
                                do_sample=True,
                                temperature=0.7,
                                top_k=50,
                                top_p=0.9,
                                repetition_penalty=1.1)

    print("Generated Outputs:")
    for out in outputs:
        print(out)


def continued_pretraining(batch_size=8):
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print("\n Task: causal_lm (continued pretraining) on "
          "datamol-io/safe-gpt UniChem molecules")

    model_dir = CHECKPOINT_DIR
    pretrained_dir = PRETRAINED_DIR

    train_text_dataset = build_pretraining_safe_gpt_dataset()

    pretrain_model = Olmo(task_type="causal_lm",
                          tokenizer_path="allenai/OLMo-7B-hf",
                          torch_dtype=dtype,
                          finetune_strategy="qlora",
                          model_dir=model_dir,
                          learning_rate=2e-4,
                          gradient_checkpointing=True,
                          skip_weight_init=True,
                          batch_size=batch_size,
                          device=torch.device("cpu"))

    pretrain_model.load_from_pretrained("allenai/OLMo-7B-hf",
                                        from_hf_checkpoint=True)

    num_gpus = torch.cuda.device_count()
    trainer = LightningTorchModel(
        model=pretrain_model,
        batch_size=batch_size,
        model_dir=model_dir,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=-1 if torch.cuda.is_available() else 1,
        strategy="ddp" if num_gpus > 1 else "auto",
        enable_progress_bar=True,
        log_every_n_steps=1)

    print(f"Starting continued pretraining on causal_lm dataset "
          f"({num_gpus} GPU(s))...")
    trainer.fit(train_text_dataset,
                nb_epoch=1,
                max_checkpoints_to_keep=1,
                num_workers=0)

    is_main_process = trainer.trainer.is_global_zero

    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    if is_main_process:
        run_generation(pretrain_model, quantized=True)

        if pretrain_model.finetune_strategy in ("lora", "qlora"):
            pretrain_model.model = pretrain_model.model.merge_and_unload()

        shutil.rmtree(pretrained_dir, ignore_errors=True)
        pretrain_model.model.save_pretrained(pretrained_dir)
        pretrain_model.tokenizer.save_pretrained(pretrained_dir)
        print(f"Backbone saved to {pretrained_dir}")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    del pretrain_model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50000,
        help="Number of unique UniChem molecules to use for continued "
        "pretraining.")
    parser.add_argument(
        "--max-smiles-length",
        type=int,
        default=200,
        help="Maximum canonical SMILES string length to include, to bound "
        "tokenized sequence length and avoid GPU OOMs from outlier "
        "molecules.")
    args = parser.parse_args()

    MAX_SAMPLES = args.max_samples
    MAX_SMILES_LENGTH = args.max_smiles_length

    continued_pretraining()