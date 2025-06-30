

import os
import re

# Define the directory where all the files are stored
root_dir = "/content/deepchem/deepchem/molnet/load_function"

# List of file extensions to process
file_extensions = ['.py']

# Function to update the code in each file
def update_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Step 1: Check if 'get_featurizer' is used and remove it
    if 'get_featurizer' in content:
        content = content.replace('get_featurizer(featurizer)', 'self.featurizer')
    
    # Step 2: Check if class has 'create_dataset' and add '__init__' if needed
    class_match = re.search(r'class\s+_([A-Za-z]+)Loader\s*\(', content)
    if class_match:
        class_name = class_match.group(1)
        if '__init__' not in content:
            # Add __init__ method if not present
            init_code = f"""
    def __init__(self, featurizer, *args, **kwargs):
        super(_{class_name}Loader, self).__init__(*args, **kwargs)
        self.featurizer = featurizer
            """
            content = content.replace(f'class _{class_name}Loader', f'class _{class_name}Loader' + init_code)
    
    # Write the updated content back to the file
    with open(file_path, 'w') as file:
        file.write(content)
    
    print(f"Updated file: {file_path}")

# Function to recursively process all files in the directory
def process_directory(root_dir):
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if any(file.endswith(ext) for ext in file_extensions):
                file_path = os.path.join(subdir, file)
                update_file(file_path)

# Run the update
process_directory(root_dir)
