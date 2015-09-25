#!/bin/bash

# Download PCBA JSON files for all AIDs into the current directory
# usage: download_pcba_json.sh [n_jobs=8]
if [ -z $1 ]
then
  n_jobs=8
else
  n_jobs=$1
fi
url=ftp://ftp.ncbi.nlm.nih.gov/pubchem/Bioassay/JSON/
echo "Downloading PCBA JSON files (${n_jobs} jobs)..."
curl -sl ${url} | awk "{print \"${url}\"\$1}" | xargs -n 1 -P ${n_jobs} wget -q
