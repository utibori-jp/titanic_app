import pandas as pd
from data_processer import preprocessing

class DataSet:
    def __init__(self):
        root_path = "rootpath/titanic/"
        self.train_data = pd.read_csv(root_path + "train.csv")
        self.test_data = pd.read_csv(root_path + "test.csv")
        self.y_test = pd.read_csv(root_path + "gender_submission.csv")
        self.train_data_for_ML = preprocessing(self.train_data)
        self.test_data_for_ML = preprocessing(self.test_data)
        self.target = "Survived"
        self.column_list = list(self.test_data.columns)
        self.column_list_processed = list(self.test_data_for_ML.columns)
