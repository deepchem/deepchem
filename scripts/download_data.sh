#!/bin/bash
# After running this script all data is in datasets. Assumes script is run from base directory
cd datasets
find ../examples -name "get*.sh" -exec bash {} \;
