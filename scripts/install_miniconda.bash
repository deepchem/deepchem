# This script is based on https://github.com/kangwonlee/travis-yml-conda-posix-nt

# distigunish OS
case $(uname | tr '[:upper:]' '[:lower:]') in
  linux*)
    export OS_NAME=linux
    ;;
  darwin*)
    export OS_NAME=osx
    ;;
  msys*)
    export OS_NAME=windows
    ;;
  *)
    export OS_NAME=notset
    ;;
esac

# prepare env variable
if [[ "$OS_NAME" != "windows" ]]; then
    export MINICONDA_PATH=$HOME/miniconda
    export MINICONDA_SUB_PATH=$MINICONDA_PATH/bin
elif [[ "$OS_NAME" == "windows" ]]; then
    export MINICONDA_PATH=$HOME/miniconda
    export MINICONDA_PATH_WIN=`cygpath --windows $MINICONDA_PATH`
    export MINICONDA_SUB_PATH=$MINICONDA_PATH/Scripts
fi

# download installer for Linux and MacOSX
if [[ "$OS_NAME" != "windows" ]]; then
    mkdir -p $HOME/miniconda_installer
    if [[ "$OS_NAME" == "linux" ]]; then
        wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $HOME/miniconda_installer/miniconda.sh
    elif [[ "$OS_NAME" == "osx" ]]; then
        wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O $HOME/miniconda_installer/miniconda.sh
    fi
fi

# install minicoda
if [[ "$OS_NAME" == "windows" ]]; then
    choco install miniconda3 --params="'/JustMe /AddToPath:1 /D:$MINICONDA_PATH_WIN'"
else
    bash $HOME/miniconda_installer/miniconda.sh -b -u -p $MINICONDA_PATH
fi

# add path
export MINICONDA_LIB_BIN_PATH=$MINICONDA_PATH/Library/bin
export PATH="$MINICONDA_PATH:$MINICONDA_SUB_PATH:$MINICONDA_LIB_BIN_PATH:$PATH"

# update conda
source $MINICONDA_PATH/etc/profile.d/conda.sh
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
