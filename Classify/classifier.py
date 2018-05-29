import pandas as pd
import numpy as np

from fuzzywuzzy import fuzz
from fuzzywuzzy import process

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def get_exact_match(data, name):
    """ Get exact match of the name from dataset """
    return data[(data['Name'] == name)]

def get_df(data_path):
    """ Returns the csv dataset as a dataframe """
    return pd.read_csv(data_path)

def get_clean_data(data):
    """ Cleans up the dataset for training and testing """
    remove_cols = ['PassengerId', 'Pclass', 'Ticket', 'Fare', 'Cabin']
    keep_cols = ['Sex', 'Age', 'Embarked']
    data = data.drop(remove_cols, axis=1)
    for cols in keep_cols:
        data[cols] = pd.Categorical(data[cols])
        data[cols] = data[cols].cat.codes
    return data.drop(['Survived', 'Name'], axis=1), pd.DataFrame(data.Survived), pd.DataFrame(data.Name)

def train_classifier(data):
    """ Random Forest Classifier model with split of dataset into 
        train and test data and testing of model accuracy """
    model = RandomForestClassifier()
    X_train, X_test, Y_train, Y_test = train_test_split(data[0], data[1], test_size=0.3)
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    acc_score = accuracy_score(Y_test, predictions)
    return predictions, acc_score

def train_classifier_up(data, test):
    """ Random Forest Classifier model with complete training data,
        testing data passed to it separately """
    model = RandomForestClassifier()
    model.fit(data[0], data[1])
    predictions = model.predict(test)
    return predictions

def fuzzy_match():
    """ 79% fuzzy match """ 
    return fuzz.token_set_ratio('Heikkinen Lain', 'Heikkinen, Miss. Laina')

def fuzzy_matcher(query, data):
    return fuzz.token_set_ratio(query, data)

def get_fuzz_name(df, query, tolerance=75):
    """ Create an additional column fuzzer which contains
        fuzzy distance between names with tolerance upto 100 """
    if 'fuzzer' in df.columns:
        df.drop('fuzzer', axis=1)
    df['fuzzer'] = [fuzzy_matcher(query, row.Name) for row in df.itertuples()]
    return df[(df['fuzzer']>=tolerance)]

