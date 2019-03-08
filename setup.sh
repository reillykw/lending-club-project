#!/usr/bin/env bash
python3.7 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
mkdir data
wget https://resources.lendingclub.com/LoanStats_2018Q4.csv.zip -O data/LoanStats_2018Q4.csv.zip &
unzip data/LoanStats_2018Q4.csv.zip -d data
