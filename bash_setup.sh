#! /bin/bash

## Starting the scipt and printing this in terminal
echo "The script starts here."

## Update packages
sudo apt-get update 

## Install pip
#curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
#python3.8 get-pip.py
sudo apt install python3-pip

## Install python-tk
sudo apt-get install python3-tk

## Download package
git clone https://github.com/Ben-Jamin-Griff/Activity-Classification-with-TensorFlow.git

## Pull data in from bucket
gsutil cp -r gs://apc_bucket_1/* ./Activity-Classification-with-TensorFlow

## Move into package
cd Activity-Classification-with-TensorFlow

## Install VENV and activate
#python3 -m pip install --user virtualenv
sudo apt install python3.8-venv
python3 -m venv venv
source venv/bin/activate

## Install requirements into VENV
pip install -r requirements.txt

## Running the script in the shell
python3 classify/shallow_paper_processing.py

## Running the script in the background (to close the shell)
nohup python3 classify/shallow_paper_processing.py &
## Check script is still running
ps -A