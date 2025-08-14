

import os
import re

ROOT_DIR = "/content/deepchem/deepchem/molnet/load_function"  # The root directory where your loader files are located

def remove_get_featurizer_import(filepath):
    """Remove `get_featurizer` import from files."""
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Search for the import statement
    import_pattern = r'from deepchem.molnet.featurizers import get_featurizer'

    # If the import exists, remove it
    new_lines = [line for line in lines if not re.match(import_pattern, line)]
    
    # If we made any changes, write back to the file
    if len(new_lines) != len(lines):
        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        print(f"Removed unused import from: {filepath}")

def walk_loader_files():
    """Walk through all Python files in the directory and remove `get_featurizer` import."""
    for dirpath, _, filenames in os.walk(ROOT_DIR):
        for filename in filenames:
            if filename.endswith(".py"):
                full_path = os.path.join(dirpath, filename)
                remove_get_featurizer_import(full_path)

if __name__ == "__main__":
    walk_loader_files()
