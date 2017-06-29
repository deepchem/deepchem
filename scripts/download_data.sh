#!/bin/bash
# After running this script all data is in datasets. Assumes script is run from deepchem/scripts 
# e.g., user@server:~/deepchem/scripts$ ./download_data.sh

cd ../datasets
find ../examples -name "get*.sh" -exec bash {} \;
