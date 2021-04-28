#! /bin/bash -e

apt-get update
apt-get install -y git wget

wget -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash miniconda.sh -b -p /conda
/conda/bin/conda update -n base conda

git clone https://github.com/alecgunny/ligo-o2-bbh-cloud.git
cd ligo-o2-bbh-cloud/client
/conda/bin/conda env create -f environment.yaml