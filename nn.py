import pandas as pd
import numpy as np
from itertools import product
from pandas import Series
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler, Normalizer, OrdinalEncoder, OneHotEncoder, FunctionTransformer
import tensorflow as tf
import numpy as np
from itertools import product
from enum import Enum
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

DATAFILE = 'data/LoanStats_20{year}Q{quarter}.csv'
dframes = list()
for year, quarter in product([16, 17, 18], [1, 2, 3, 4]):
    temp_df = pd.read_csv(DATAFILE.format(year=year, quarter=quarter),
                            sep=",", skiprows=1, low_memory=False)# nrows=10000)
    dframes.append(temp_df)
loan_data = pd.concat(dframes, axis=0, ignore_index=True)

columns = [
"loan_amnt",
"int_rate",
"grade",
"sub_grade",
"emp_length",
"home_ownership",
"annual_inc",
"verification_status",
"loan_status",
"dti",
"open_acc",
"pub_rec",
"revol_bal",
"revol_util",
"total_acc",
]
loan_data = loan_data[columns]
loan_data['revol_util'] = loan_data['revol_util'].str.replace("%", "").astype(float) / 100
loan_data['int_rate'] = loan_data['int_rate'].str.replace("%", "").astype(float) / 100
loan_data = loan_data[np.isfinite(loan_data['loan_amnt'])]
loan_data.groupby(['emp_length']).count()
loan_data = loan_data[loan_data.loan_status.isin(['Fully Paid', 'Charged Off'])]

loan_data = loan_data.dropna()

def emp_length(x: str):
    emps = ['10+ years', '< 1 year', '1 year', '2 years', '3 years', '4 years',
           '5 years','6 years', '7 years', '8 years', '9 years', 'placeholder','10+ years']
    try:
        i = emps.index(x)
        return i
    except:
        return -1

loan_data.emp_length = loan_data.emp_length.apply(emp_length)

normalizers = ['emp_length', 'int_rate', 'loan_amnt', 'annual_inc', 'dti', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc']
ordinals = ['grade', 'sub_grade', 'loan_status']
one_hot = ['home_ownership', 'verification_status']
for ordinal in ordinals + one_hot:
    loan_data[ordinal] = LabelEncoder().fit_transform(loan_data[ordinal].values.reshape(-1,1))
for norm in normalizers:
    loan_data[norm] = StandardScaler().fit_transform(loan_data[norm].values.reshape(-1,1))

loan_data['comb'] = loan_data['grade'] * loan_data['sub_grade']

# Vary the depth from {3, 5, 9} and width from {5, 10, 25, 50, 100}


class Activation(Enum):
    relu = tf.nn.relu
    tanh = tf.nn.tanh


def setup(depth: int, width: int, activ_fun: Activation):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    for i in range(int(depth)):
        model.add(tf.keras.layers.Dense(
            width
            , activation=tf.nn.softmax
            , kernel_initializer=tf.keras.initializers.glorot_uniform() if activ_fun == Activation.tanh else tf.keras.initializers.he_normal()
        ))
    model.add(tf.keras.layers.Dense(2,
                                    activation=tf.nn.softmax))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'],
                  )
    return model


if __name__ == "__main__":
    depth = np.linspace(6, 9, 3)
    width = np.linspace(5, 100, 20)
    train_err = np.zeros([len(depth), len(width)])
    test_err = np.zeros([len(depth), len(width)])
    Y = loan_data.loan_status.values
    X = loan_data.drop('loan_status', 1).values
    train_x = X
    train_y = Y
    i = 0
    print('Starting')
    for activ_fun in [Activation.relu, Activation.relu]:
        print(activ_fun)
        for dw in product(depth, width):
            model = setup(dw[0], dw[1], activ_fun)
            output = model.fit(train_x, train_y, epochs=1)
            #test_loss, test_acc = model.evaluate(test_data[[1, 2, 3, 4]].values, test_data[4].values)
            train_err[int(dw[0] / 3 - 1), int(dw[1] / 5 - 1)] = 1 - output.history['acc'][0]
            #test_err[int(dw[0] / 3 - 1), int(dw[1] / 5 - 1)] = 1 - test_acc
            #train_err[dw[1]][dw[0]] = 1 - output.history['accuracy'][2]
            i += 1
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        #Z = np.g
        X, Y = np.meshgrid(depth, width)
        ax.plot_wireframe(X, Y, train_err.T, color='blue', label='train')
        #ax.plot_wireframe(X, Y, test_err.T, color='red', label='test')
        ax.view_init(30, 120)
        ax.set_xlabel("depth")
        ax.set_ylabel("# neurons")
        ax.set_zlabel("error")
        ax.legend()
        ax.set_title('tanh' if activ_fun == Activation.tanh else 'relu')
        plt.show()