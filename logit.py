
import pickle
from sklearn.model_selection import train_test_split
import math
from pprint import PrettyPrinter
import tensorflow as tf
import numpy as np
from itertools import product
from lib.utils import train
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.ensemble import RandomForestClassifier


pp = PrettyPrinter()

print("Retrieve Data...")
with open('X.dat', 'rb') as xr:
    X = pickle.load(xr)
with open('Y.dat', 'rb') as yr:
    Y = pickle.load(yr)
print("Data Retrieved...")
#
# X = X[:10000, :]
# Y = Y[:10000]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=.9)

print("Logistic Regression")
cls = LogisticRegression(solver='newton-cg')
results = train(cls, X_train, X_test, Y_train, Y_test)
pp.pprint(results)
print("Perceptron")
cls = Perceptron(penalty='l2')
results = train(cls, X_train, X_test, Y_train, Y_test)
pp.pprint(results)
print("Random Forest")
cls = RandomForestClassifier(n_estimators=100)
results = train(cls, X_train, X_test, Y_train, Y_test)
pp.pprint(results)
print("Neural Net")
delta_depth = 3
delta_width = 6
depth = np.linspace(3, 9, delta_depth, endpoint=False)
width = np.linspace(5, 100, delta_width, endpoint=False)
train_err = np.zeros([len(depth), len(width)])
test_err = np.zeros([len(depth), len(width)])
for dw in product(depth, width):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    for _ in range(int(dw[0])):
        model.add(tf.keras.layers.Dense(
            int(dw[1])
            , activation=tf.nn.relu
            ,
            kernel_initializer=tf.keras.initializers.he_uniform()
        ))
    model.add(tf.keras.layers.Dense(3,
                                    activation=tf.nn.softmax))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy', ],# accuracy_score, precision_score, recall_score, f1_score],
                  )

    output = model.fit(X_train, Y_train.values, epochs=5)

    # train = dict()
    # train['fully_paid'] = scores(Y_train.values, y_train, pos_label=1)
    # train["default"] = scores(Y_train.values, y_train, pos_label=0)
    # test = dict()
    # test["fully_paid"] = scores(Y_test.values, y_test, pos_label=1)
    # test["default"] = scores(Y_test.values, y_test, pos_label=0)
    # pp.pprint(dict(test=test, train=train))
    test_loss, test_acc = model.evaluate(X_test, Y_test)
    index_row = math.floor(((dw[0] - 3) / ((9 - 3) / delta_depth)))
    index_col = math.floor(((dw[1] - 5) / ((100 - 5) / delta_width)))
    print(index_row, index_col, dw[0], dw[1])
    test_err[index_row, index_col] = 1 - test_acc
    train_err[index_row, index_col] = 1 - output.history['acc'][0]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Z = np.g
X, Y = np.meshgrid(depth, width)
ax.plot_wireframe(X, Y, train_err.T, color='blue', label='train')
ax.plot_wireframe(X, Y, test_err.T, color='red', label='test')
ax.view_init(30, 120)
ax.set_xlabel("depth")
ax.set_ylabel("# neurons")
ax.set_zlabel("error")
ax.legend()
ax.set_title('relu')
plt.show()

