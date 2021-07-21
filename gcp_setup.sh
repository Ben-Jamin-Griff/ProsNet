#! /bin/bash

## Starting the scipt and printing this in terminal
echo "The script starts here."

## Install pip

## Install python-tk
sudo apt-get install python3-tk

## Download package
git clone https://github.com/Ben-Jamin-Griff/Activity-Classification-with-TensorFlow.git

## Pull data in from bucket
gsutil cp -r gs://<my-bucket>/* /Activity-Classification-with-TensorFlow

## Move into package
cd Activity-Classification-with-TensorFlow

## Install VENV and activate
python3 -m pip install --user virtualenv
python3 -m venv env
source venv/bin/activate

## Install requirements into VENV
pip install -r requirements.txt

## Install tk using pip (not sure if I need this?)
#pip install tk

## Running the script
python3 classify/shallow_paper_processing.py