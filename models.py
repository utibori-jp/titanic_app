from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

classifiers = {
    "LogisticRegression":LogisticRegression(),
    "KNeighborsClassifier":KNeighborsClassifier(),
    "DecisionTreeClassifier":DecisionTreeClassifier(),
    "RandomForestClassifier":RandomForestClassifier(),
    "SVC":SVC(),
    "XGBClassifier":XGBClassifier()
}

def get_predict_score(train_data, test_data, y_test, classifier):
    target = "Survived"
    X_train = train_data.drop(target, axis = 1)
    y_train = train_data[target]

    model = classifiers[classifier]
    model.fit(X_train, y_train)

    pred = model.predict(test_data)
    a = y_test.loc[y_test[target] == pred]

    score_dict = {
        "train_score":model.score(X_train, y_train),
        "test_score":len(a)/len(y_test[target]),
    }
    return score_dict
