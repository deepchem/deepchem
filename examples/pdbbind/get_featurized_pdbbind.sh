echo "Pulling featurized core pdbbind dataset from deepchem"
wget -c http://deepchem.io.s3-website-us-west-1.amazonaws.com/featurized_datasets/core_grid.tar.gz
echo "Extracting core pdbbind"
tar -zxvf core_grid.tar.gz
echo "Pulling featurized refined pdbbind dataset from deepchem"
wget -c http://deepchem.io.s3-website-us-west-1.amazonaws.com/featurized_datasets/refined_grid.tar.gz
echo "Extracting refined pdbbind"
tar -zxvf refined_grid.tar.gz
echo "Pulling featurized full pdbbind dataset from deepchem"
wget -c http://deepchem.io.s3-website-us-west-1.amazonaws.com/featurized_datasets/full_grid.tar.gz
echo "Extracting full pdbbind"
tar -zxvf full_grid.tar.gz
