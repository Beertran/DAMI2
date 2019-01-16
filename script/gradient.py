import numpy as np
import pandas as pd

train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test  = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
mark="Survived"


def harmonize_data(titanic):
    
    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
    titanic["Age"].median()
    
    
    
    titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
    titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
    
    titanic["Embarked"] = titanic["Embarked"].fillna("S")

    titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
    titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
    titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())

    return titanic


def create_submission(clf, train, test, predictors, filename):

    clf.fit(train[predictors], train["Survived"])
    predictions = clf.predict(test[predictors])

    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions
    })
    
    submission.to_csv(filename, index=False)


from sklearn.model_selection import cross_val_score

def validation_scores(clf, train_data):
    scores = cross_val_score(
        clf,
        train_data[predictors],
        train_data[mark],
        cv=3
    )
    return scores.mean()


train_data = harmonize_data(train)
test_data  = harmonize_data(test)


def compare_metods(classifiers, train_data):
    names, scores = [], []
    for name, clf in classifiers:
        names.append(name)
        scores.append(validation_scores(clf, train_data))
    return pd.DataFrame(scores, index=names, columns=['Scores'])



from sklearn.ensemble import GradientBoostingClassifier

classifiers = [
    ("Gradient Boosting", GradientBoostingClassifier(max_depth=4)),
]

res = compare_metods(classifiers, train_data)
res

print(res)
