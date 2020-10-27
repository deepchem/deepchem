# This script creates the new deepchem enviroment

CMDNAME=`basename $BASH_SOURCE`
if [ $# -ne 2 ]; then
    echo "Please set two arguments."
    echo "Usage) source $CMDNAME python_version cpu_or_gpu" 1>&2
    echo "Example) source $CMDNAME 3.6 gpu" 1>&2
    return 1
fi

# This command is nearly equal to `conda init` command
# Need to use `conda activate` command
eval "$(conda shell.bash hook)"

# Create deepchem environment
conda config --set always_yes yes
conda create --name deepchem python=$1
conda install -c conda-forge conda-merge

if [ "$2" = "gpu" ];
then
    # We expect the CUDA vesion is 10.1.
    conda-merge $PWD/env.common.yml $PWD/env.gpu.yml $PWD/env.test.yml > $PWD/env.yml
    echo "Installing DeepChem in the GPU environment"
else
    if [ "$(uname)" == 'Darwin' ];
    then
        conda-merge $PWD/env.common.yml $PWD/env.cpu.mac.yml $PWD/env.test.yml > $PWD/env.yml
    else
        conda-merge $PWD/env.common.yml $PWD/env.cpu.yml $PWD/env.test.yml > $PWD/env.yml
    fi
    echo "Installing DeepChem in the CPU environment"
fi

# Install all dependencies
conda env update --file $PWD/env.yml
