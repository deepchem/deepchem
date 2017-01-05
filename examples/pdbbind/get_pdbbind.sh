echo "Pulling pdbbind dataset from deepchem"
wget -c http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/pdbbind_v2015.tar.gz
echo "Extracting pdbbind structures"
tar -zxvf pdbbind_v2015.tar.gz
