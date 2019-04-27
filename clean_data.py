from lib.utils import get_train_data
import pickle

print("Retrieve Data...")
X, Y = get_train_data()
with open('X.dat', 'wb') as xb:
    pickle.dump(X, xb)
with open('Y.dat', 'wb') as yb:
    pickle.dump(Y, yb)
