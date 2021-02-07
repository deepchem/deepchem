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

$common = Join-Path (pwd).PATH "env.common.yml"
$test = Join-Path (pwd).PATH "env.test.yml"
$out = Join-Path (pwd).PATH "env.yml"
if($args[1] -eq "gpu")
{
    # We expect the CUDA vesion is 10.1.
    $gpu = Join-Path (pwd).PATH "env.gpu.yml"
    conda-merge $common $gpu $test > $out
    echo "Installing DeepChem in the GPU environment"
}
else
{
    $cpu = Join-Path (pwd).PATH "env.cpu.yml"
    conda-merge $common $cpu $test > $out
    echo "Installing DeepChem in the CPU environment"
}

# Install all dependencies
conda env update --file $out
