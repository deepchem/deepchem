echo "Pulling featurized and split ACNN datasets from deepchem"
wget http://deepchem.io.s3-website-us-west-1.amazonaws.com/featurized_datasets/acnn_refined.tar.gz
echo "Extracting ACNN datasets"
tar -zxvf acnn_refined.tar.gz
