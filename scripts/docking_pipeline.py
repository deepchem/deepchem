import argparse
import json
import logging
import os
from typing import Iterable, List, Optional, Tuple

import deepchem as dc
from deepchem.dock import Docker, VinaPoseGenerator

# -----------------------------
# Logging
# -----------------------------
logger = logging.getLogger("docking_pipeline")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# -----------------------------
# Utils
# -----------------------------
def validate_inputs(protein_file: str, ligand_file: str) -> None:
    if not os.path.isfile(protein_file):
        raise FileNotFoundError(f"Protein file not found: {protein_file}")
    if not os.path.isfile(ligand_file):
        raise FileNotFoundError(f"Ligand file not found: {ligand_file}")


def ensure_dir(path: Optional[str]) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


# -----------------------------
# Core Docking
# -----------------------------
def run_docking(
    protein_file: str,
    ligand_file: str,
    exhaustiveness: int = 8,
    num_modes: int = 9,
    score_with_pose_generator: bool = True,
) -> List[Tuple[object, float]]:
    """
    Run docking using DeepChem Vina backend.

    Returns:
        List of tuples (pose, score)
    """
    # Graceful dependency check for vina
    try:
        from vina import Vina  # noqa: F401
    except ImportError:
        logger.error("AutoDock Vina is not installed.")
        logger.error(
            "Install it using one of the following methods:\n"
            "1) conda install -c conda-forge vina\n"
            "2) brew install vina (if available via tap)\n"
            "3) Build from source: https://github.com/ccsb-scripps/AutoDock-Vina"
        )
        raise ImportError("Vina dependency missing")

    logger.info("Initializing Vina pose generator...")
    pose_gen = VinaPoseGenerator(pocket_finder=None)

    logger.info("Creating Docker...")
    docker = Docker(pose_gen)

    logger.info(
        "Running docking | exhaustiveness=%d, num_modes=%d",
        exhaustiveness,
        num_modes,
    )

    try:
        results = list(
            docker.dock(
                (protein_file, ligand_file),
                exhaustiveness=exhaustiveness,
                num_modes=num_modes,
                use_pose_generator_scores=score_with_pose_generator,
            )
        )
    except Exception as e:
        logger.exception("Docking engine failed. Ensure Vina is correctly installed.")
        raise

    logger.info("Docking completed. Generated %d poses.", len(results))
    return results


# -----------------------------
# Output Handlers
# -----------------------------
def save_scores(
    results: Iterable[Tuple[object, float]],
    out_file: str,
) -> None:
    logger.info("Saving scores to %s", out_file)
    serializable = [
        {"pose_index": i, "score": float(score)}
        for i, (_, score) in enumerate(results)
    ]
    with open(out_file, "w") as f:
        json.dump(serializable, f, indent=2)


def print_top_k(
    results: List[Tuple[object, float]],
    k: int = 5,
) -> None:
    logger.info("Top %d poses:", k)
    for i, (_, score) in enumerate(results[:k]):
        print(f"[{i}] score = {score:.4f}")


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DeepChem Vina Docking Pipeline"
    )
    parser.add_argument("--protein", required=True, help="Protein PDB file")
    parser.add_argument("--ligand", required=True, help="Ligand SDF/SMI file")
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--num_modes", type=int, default=9)
    parser.add_argument("--out", type=str, default=None, help="Output JSON path")
    parser.add_argument("--top_k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    validate_inputs(args.protein, args.ligand)
    if args.out:
        ensure_dir(os.path.dirname(args.out))

    try:
        results = run_docking(
            protein_file=args.protein,
            ligand_file=args.ligand,
            exhaustiveness=args.exhaustiveness,
            num_modes=args.num_modes,
        )

        if not results:
            logger.warning("No docking poses generated.")
            return

        print_top_k(results, k=args.top_k)

        if args.out:
            save_scores(results, args.out)

    except Exception as e:
        logger.exception("Docking failed due to runtime/dependency error: %s", str(e))
        raise


if __name__ == "__main__":
    main()