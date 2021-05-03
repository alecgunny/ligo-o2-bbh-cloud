#! /bin/bash -e

cmd="$1"
shift

if [[ "$cmd" == "install" ]]; then
    wget -q -O $HOME/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash $HOME/miniconda.sh -b -p $HOME/conda
    $HOME/conda/bin/conda update -q -n base -y conda
    $HOME/conda/bin/conda init
    rm $HOME/miniconda.sh
elif [[ "$cmd" == "create" ]]; then
    $HOME/conda/bin/conda env create -yq -f environment.yaml
elif [[ "$cmd" == "run" ]]; then
    $HOME/conda/bin/conda activate o2
    python client.py "$@"
fi