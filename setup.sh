#!/usr/bin/env bash
python3.7 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
mkdir data
years=(16 17 18)
for y in ${years[*]};
do
    for q in {1..4}
    do
        wget https://resources.lendingclub.com/LoanStats_20${y}Q${q}.csv.zip -O data/LoanStats_20${y}Q${q}.csv.zip
        unzip data/LoanStats_20${y}Q${q}.csv.zip -d data
    done
done
for l in {a..d}
do
    wget https://resources.lendingclub.com/LoanStats3${l}.csv.zip -O data/LoanStats3${l}.csv.zip
    unzip data/LoanStats3${l}.csv.zip -d data
done
rm data/*.zip
