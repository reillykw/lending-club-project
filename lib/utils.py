from lib import data_cols
import numpy as np
import pandas as pd
from itertools import product
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def emp_length(x: str):
    """
    Transformer for employment length

    :param x: employment length string

    :return:
    """
    emps = ['< 1 year', '1 year', '2 years', '3 years', '4 years',
           '5 years', '6 years', '7 years', '8 years', '9 years', 'placeholder', '10+ years']
    try:
        i = emps.index(x)
        return i
    except:
        return -1


def loan_status(status: str):
    """
    Transformer for loan status

    :param status: loan status

    :return:
    """
    if status == 'Fully Paid':
        return 1
    else:
        return 0


def get_train_data(nrows=None):
    """
    Loads the loan stat data from file.

    Note: If data is not present, run ./setup.sh script to download from source.

    :return: dataframe
    """
    DATAFILE = 'data/LoanStats_20{year}Q{quarter}.csv'
    OLDER_DATAFILES = 'data/LoanStats3{letter}.csv'
    dframes = list()
    log.info("Reading data from files...")
    for year, quarter in product([16, 17, 18], [1, 2, 3, 4]):
        temp_df = pd.read_csv(DATAFILE.format(year=year, quarter=quarter),
                              sep=",", skiprows=1, nrows=nrows, low_memory=False)
        dframes.append(temp_df)
    for letter in ['a', 'b', 'c', 'd']:
        temp_df = pd.read_csv(OLDER_DATAFILES.format(letter=letter),
                              sep=",", skiprows=1, nrows=nrows, low_memory=False)
        dframes.append(temp_df)
    log.info("Appending data...")
    loan_data = pd.concat(dframes, axis=0, ignore_index=True)
    log.info("Cleaning Data")
    # Remove unwanted columns, clean out unwanted data
    loan_data = loan_data[data_cols]
    X, Y = clean_data(loan_data)
    return X, Y


def clean_data(loan_data: pd.DataFrame) -> pd.DataFrame:
    """
    Quick cleaning of the data, very customized
    :param loan_data: dataframe of loans
    :return: X, Y
    """
    perc_cols = ['revol_util', 'int_rate']
    for col in perc_cols:
        if col in loan_data.columns:
            loan_data[col] = loan_data[col].str.replace("%", "").astype(float) / 100
    # Only finite loans that are paid off or defaulted
    loans = loan_data[np.isfinite(loan_data['loan_amnt'])]
    del loan_data
    loans.emp_length = loans.emp_length.apply(emp_length)
    loans = loans[loans.loan_status.isin(['Fully Paid', 'Charged Off', 'Default'])]
    loans = remove_missing_columns(loans)
    loans = loans.dropna()
    grades = OneHotEncoder().fit_transform(loans.grade.values.reshape(-1, 1))
    states = OneHotEncoder().fit_transform(loans.addr_state.values.reshape(-1, 1))
    zips = OneHotEncoder().fit_transform(loans.zip_code.values.reshape(-1, 1))
    purpose = OneHotEncoder().fit_transform(loans.purpose.values.reshape(-1, 1))
    ho = OneHotEncoder().fit_transform(loans.home_ownership.values.reshape(-1, 1))
    application_type = OneHotEncoder().fit_transform(loans.application_type.values.reshape(-1, 1))
    verify = OneHotEncoder().fit_transform(loans.verification_status.values.reshape(-1, 1))
    sub_grade = OneHotEncoder().fit_transform(loans.sub_grade.values.reshape(-1, 1))
    Y = loans['loan_status'].apply(loan_status)
    del loans['loan_status']
    del loans['addr_state']
    del loans['zip_code']
    del loans['purpose']
    del loans['home_ownership']
    del loans['grade']
    del loans['emp_length']
    del loans['verification_status']
    del loans['application_type']
    del loans['sub_grade']
    data = preprocessing.scale(loans)
    del loans
    data = np.append(data, grades.toarray(), axis=1)
    data = np.append(data, states.toarray(), axis=1)
    data = np.append(data, zips.toarray(), axis=1)
    data = np.append(data, ho.toarray(), axis=1)
    data = np.append(data, purpose.toarray(), axis=1)
    data = np.append(data, application_type.toarray(), axis=1)
    data = np.append(data, verify.toarray(), axis=1)
    data = np.append(data, sub_grade.toarray(), axis=1)
    return data, Y


def scores(y_true, y_pred, pos_label: int = 1) -> list:
    r_score = recall_score(y_true=y_true, y_pred=y_pred, pos_label=pos_label)
    f_score = f1_score(y_true=y_true, y_pred=y_pred, pos_label=pos_label)
    p_score = precision_score(y_true=y_true, y_pred=y_pred, pos_label=pos_label)
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    return dict(accuracy=accuracy, precision=p_score, recall=r_score, f1=f_score)


def train(cls, X_train, X_test, Y_train, Y_test):
    print("Classifier Training...")
    cls.fit(X_train, Y_train)

    y_train = cls.predict(X_train)
    y_test = cls.predict(X_test)
    train = dict()
    train['fully_paid'] = scores(Y_train, y_train, pos_label=1)
    train["default"] = scores(Y_train, y_train, pos_label=0)
    test = dict()
    test["fully_paid"] = scores(Y_test, y_test, pos_label=1)
    test["default"] = scores(Y_test, y_test, pos_label=0)
    return dict(test=test, train=train)


def remove_missing_columns(loan_data: pd.DataFrame) -> pd.DataFrame:
    percent_missing = loan_data.isnull().sum() * 100 / len(loan_data)
    missing_value_df = pd.DataFrame({'column_name': loan_data.columns,
                                     'percent_missing': percent_missing})
    return loan_data.drop(missing_value_df[missing_value_df.percent_missing > 10].column_name, axis=1)
