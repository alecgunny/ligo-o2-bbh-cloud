#! /bin/bash -e

wget -O $HOME/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash $HOME/miniconda.sh -b -p $HOME/conda
$HOME/conda/bin/conda update -n base conda
$HOME/conda init
rm $HOME/miniconda.sh
