import pandas as pd
from sklearn.preprocessing import LabelEncoder

def fill_null_values(data):
    data.fillna({"Age": data["Age"].median()}, inplace=True)
    data.fillna({"Cabin": data["Cabin"].mode()[0]}, inplace=True)
    data.fillna({"Fare": data["Fare"].median()}, inplace=True)
    data.fillna({"Embarked": data["Embarked"].mode()[0]}, inplace=True)
    return data

def encoding_categorical_columns(data):
    data["Sex"] = LabelEncoder().fit_transform(data["Sex"])
    data = pd.get_dummies(data, columns = ["Pclass", "Embarked"])
    return data

def remove_unneccessary_columns(data):
    data.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)
    return data

def preprocessing(data):
    data = fill_null_values(data)
    data = encoding_categorical_columns(data)
    data = remove_unneccessary_columns(data)
    return data
