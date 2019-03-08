import pandas as pd
from pandas import Series
from scipy.stats import entropy
from collections import Counter
from pprint import PrettyPrinter
from multiprocessing import Pool

MAX_PROCESSORS = 10

pp = PrettyPrinter()

DATAFILE = 'data/LoanStats_2018Q4.csv'
loan_data = pd.read_csv(DATAFILE, sep=",", skiprows=1, nrows=10000)
entropies = dict()
results = dict()
pool = Pool(MAX_PROCESSORS)


def calc_entropy(column):
    counter = Counter(column)
    return column, entropy(list(counter.values()))


if __name__ == "__main__":
    for column in loan_data.columns:
        result = pool.apply_async(Counter, [loan_data[column]])
        results[column] = result

    for column, result in results.items():
        entr = result.get()
        entropies[column] = entropy(list(entr.values()))
    print(pp.pprint(entropies))
