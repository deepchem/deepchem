# This script creates the new deepchem enviroment

$CMDNAME = $myInvocation.MyCommand.name
if ($args.Count -ne 2)
{
    echo "Please set two arguments."
    echo "Usage) $CMDNAME python_version cpu_or_gpu"
    echo "Example) $CMDNAME 3.6 gpu"
    return 1
}

# This command is nearly equal to `conda init` command
# Need to use `conda activate` command
(& "conda" "shell.powershell" "hook") | Out-String | Invoke-Expression

# create deepchem environment
conda config --set always_yes yes
conda create --name deepchem python=$args[0]
conda install -c conda-forge conda-merge

$common = Join-Path (pwd).PATH "requirements/env_common.yml"
$test = Join-Path (pwd).PATH "requirements/env_test.yml"
$tensorflow = Join-Path (pwd).PATH "requirements/tensorflow/env_tensorflow.cpu.yml"
$out = Join-Path (pwd).PATH "env.yml"
# Tensorflow has same installation commands for CPU and GPU
# Torch has different installation commands for CPU and GPU
# Jax is not supported in windows. Hence, excluded.
if($args[1] -eq "gpu")
{
    # We expect the CUDA vesion is 10.1.
    $torch_gpu = Join-Path (pwd).PATH "requirements/torch/env_torch.gpu.yml"
    conda-merge $common $tensorflow $torch_gpu $test > $out
    echo "Installing DeepChem in the GPU environment"
}
else
{
    $torch_cpu = Join-Path (pwd).PATH "requirements/torch/env_torch.cpu.yml"
    conda-merge $common $tensorflow $torch_cpu $test > $out
    echo "Installing DeepChem in the CPU environment"
}

# Install all dependencies
conda env update --file $out
