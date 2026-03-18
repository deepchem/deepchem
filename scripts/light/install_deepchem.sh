# This script creates the new deepchem enviroment
# This script works on only Bash and Zsh

CMDNAME=`basename ${BASH_SOURCE:-$0}`
if [ $# -ne 3 ]; then
    echo "Please set two arguments."
    echo "Usage) source $CMDNAME python_version cpu_or_gpu" 1>&2
    echo "Example) source $CMDNAME 3.10 gpu tensorflow" 1>&2
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
    if [ "$3" = "tensorflow" ];
    then
        conda-merge $PWD/requirements/tensorflow/env_tensorflow.yml $PWD/requirements/env_test.yml > $PWD/env.yml
        echo "Installing Tensorflow environment with GPU"
    elif [ "$3" = "torch" ];
    then
        conda-merge $PWD/requirements/pytorch/env_pytorch.yml $PWD/requirements/pytorch/env_pytorch.gpu.yml $PWD/requirements/env_test.yml > $PWD/env.yml
        echo "Installing pytorch environment with GPU"
    elif [ "$3" = "jax" ];
    then
        conda-merge $PWD/requirements/jax/env_jax.yml $PWD/requirements/jax/env_jax.gpu.yml $PWD/requirements/env_test.yml > $PWD/env.yml
        echo "Installing jax environment with GPU"
    else
        conda-merge $PWD/requirements/env_common.yml $PWD/requirements/env_test.yml > $PWD/env.yml
        echo "Installing common environment with GPU"
    fi
else
    # We expect the CUDA vesion is 10.1.
    if [ "$3" = "tensorflow" ];
    then
        conda-merge $PWD/requirements/tensorflow/env_tensorflow.yml $PWD/requirements/env_test.yml > $PWD/env.yml
        echo "Installing Tensorflow environment with CPU"
    elif [ "$3" = "torch" ];
    then
        conda-merge $PWD/requirements/pytorch/env_pytorch.yml $PWD/requirements/pytorch/env_pytorch.cpu.yml $PWD/requirements/env_test.yml > $PWD/env.yml
        echo "Installing pytorch environment with CPU"
    elif [ "$3" = "jax" ];
    then
        conda-merge $PWD/requirements/jax/env_jax.yml $PWD/requirements/jax/env_jax.cpu.yml $PWD/requirements/env_test.yml > $PWD/env.yml
        echo "Installing jax environment with CPU"
    else
        conda-merge $PWD/requirements/env_common.yml $PWD/requirements/env_test.yml > $PWD/env.yml
        echo "Installing common environment with CPU"
    fi
fi
# Install all dependencies
conda env update --file $PWD/env.yml
