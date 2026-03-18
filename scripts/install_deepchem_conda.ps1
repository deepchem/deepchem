# This script creates the new deepchem enviroment


$CMDNAME = $myInvocation.MyCommand.name
if ($args.Count -ne 2)
{
    echo "Please set two arguments."
    echo "Usage) $CMDNAME python_version cpu_or_gpu"
    echo "Example) $CMDNAME 3.10 gpu"
    return 1
}

# This command is nearly equal to `conda init` command
# Need to use `conda activate` command
(& "conda" "shell.powershell" "hook") | Out-String | Invoke-Expression
$Python_version = $args[0]
$Type = $args[1]
# create deepchem environment
conda config --set always_yes yes
conda create --name deepchem python=$Python_version
conda activate deepchem
conda install -c conda-forge conda-merge

$common = Join-Path (pwd).PATH "requirements/env_common.yml"
$test = Join-Path (pwd).PATH "requirements/env_test.yml"
$out = Join-Path (pwd).PATH "env.yml"
# Torch has different installation commands for CPU and GPU
# Jax is not supported in windows. Hence, excluded.
# Tensorflow is not supported in windows. Hence, excluded, install 
# linux version using Windows Subsystem for Linux.
if($Type -eq "gpu")
{
    # We expect the CUDA vesion is 10.1.
    $torch_gpu = Join-Path (pwd).PATH "requirements/torch/env_torch.gpu.yml"
    conda-merge $common $torch_gpu $test > $out
    echo "Installing DeepChem in the GPU environment"
}
else
{
    $torch_cpu = Join-Path (pwd).PATH "requirements/torch/env_torch.win.cpu.yml"
    conda-merge $common $torch_cpu $test > $out
    echo "Installing DeepChem in the CPU environment"
}

# Install all dependencies
conda env update --file $out
