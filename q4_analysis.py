import pandas as pd
from scipy.stats import entropy
from collections import Counter
from pprint import PrettyPrinter

pp = PrettyPrinter()

DATAFILE = 'data/LoanStats_2018Q4.csv'

loan_data = pd.read_csv(DATAFILE, sep=",", skiprows=1, nrows=10000)

entropies = dict()

for column in loan_data.columns:
    print(f'{column} ', end="")
    counter = Counter(loan_data[column])
    entropies[column] = entropy(list(counter.values()))

print()
print(pp.pprint(entropies))