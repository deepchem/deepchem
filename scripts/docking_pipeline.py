

#!/usr/bin/env python3

import argparse
import json
import logging
import os

import deepchem as dc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_inputs(protein_file: str, ligand_file: str):
    if not os.path.exists(protein_file):
        raise FileNotFoundError(f"Protein file not found: {protein_file}")
    if not os.path.exists(ligand_file):
        raise FileNotFoundError(f"Ligand file not found: {ligand_file}")


def run_docking(protein_file: str, ligand_file: str, exhaustiveness=8, num_modes=9):
    # Check for vina dependency
    try:
        from vina import Vina  # noqa: F401
    except ImportError:
        logger.error("AutoDock Vina not installed. Skipping docking.")
        return []

    logger.info("Initializing Vina pose generator...")
    pose_gen = dc.dock.VinaPoseGenerator(pocket_finder=None)

    docker = dc.dock.Docker(pose_gen)

    logger.info("Running docking...")
    try:
        results = list(
            docker.dock(
                (protein_file, ligand_file),
                exhaustiveness=exhaustiveness,
                num_modes=num_modes,
                use_pose_generator_scores=True,
            )
        )
    except Exception as e:
        logger.exception("Docking failed: %s", str(e))
        return []

    logger.info("Docking completed with %d poses", len(results))
    return results


def save_results(results, output_file):
    data = []
    for i, (pose, score) in enumerate(results):
        data.append({
            "pose_id": i,
            "score": float(score) if score is not None else None
        })

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    logger.info("Results saved to %s", output_file)


def main():
    parser = argparse.ArgumentParser(description="DeepChem Docking Pipeline")
    parser.add_argument("--protein", required=True, help="Path to protein PDB file")
    parser.add_argument("--ligand", required=True, help="Path to ligand SDF file")
    parser.add_argument("--out", default="results.json", help="Output JSON file")

    args = parser.parse_args()

    validate_inputs(args.protein, args.ligand)

    results = run_docking(args.protein, args.ligand)

    if not results:
        logger.warning("No docking results generated.")
    else:
        save_results(results, args.out)


if __name__ == "__main__":
    main()