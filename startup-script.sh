#! /bin/bash -e

apt-get update
apt-get install -y git python3-pip screen
pip3 install --upgrade pip

git clone https://github.com/alecgunny/benchmark-ligo-py
cd benchmark-ligo-py
python3 -m pip install -r requirements.txt

echo "Done!"