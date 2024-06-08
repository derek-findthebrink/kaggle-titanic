# %%
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# TODO: Figure out proper way to treat dataset reassignment in python, without using class instance variables

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# import autosklearn.classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import ipdb


# Setup
# ----------------------------------------------------------------------
def setup():
    pd.set_option("display.max_row", 1000)
    pd.set_option("display.max_columns", 50)


def get_training_data():
    return pd.read_csv("./data/train.csv")


def get_test_data():
    return pd.read_csv("./data/test.csv")


def get_concatenated_data():
    train = get_training_data()
    test = get_test_data()
    df = pd.concat([train, test], axis=0, sort=True)
    return df


# Feature engineering
# ----------------------------------------------------------------------
def create_title_feature(dataset):
    # Targets are: Mr, Mrs, Master, Ms, Miss
    # Must remove entries where key === value because replace function complains
    #  - alternative is to use map, however using map resulted in a few NaN values
    #    persisting in the frame for this feature
    mapping = {
        "Capt": "Mr",
        "Col": "Mr",
        "Countess": "Mrs",
        # problematic, best guess is there were more men as doctors back then
        # => !should be computed from dataset!
        "Dr": "Mr",
        "Don": "Mr",
        "Dona": "Mrs",
        "Jonkheer": "Mr",
        "Lady": "Mrs",
        "Major": "Mr",
        # 'Master'      : 'Master',
        # 'Miss'        : 'Miss',
        "Mlle": "Miss",
        "Mme": "Mrs",
        # 'Mr'          : 'Mr',
        # 'Mrs'         : 'Mrs',
        # 'Ms'          : 'Ms',
        "Rev": "Mr",
        "Sir": "Mr",
    }

    # extract Title feature (intermediate) from name feature
    dataset["Title"] = dataset["Name"].str.extract("([A-Za-z]+)\.", expand=True)
    dataset.replace({"Title": mapping}, inplace=True)
    return dataset


def family_group(size):
    a = ""
    if size == 1:
        a = "alone"
    elif size <= 4:
        a = "small"
    else:
        a = "large"
    return a


def create_family_size_feature(dataset):
    dataset["Family_Size"] = dataset["Parch"] + dataset["SibSp"] + 1
    dataset["Family_Category"] = dataset["Family_Size"].map(family_group)
    return dataset


def create_is_alone_feature(dataset):
    dataset["Is_Alone"] = dataset["Family_Size"] == 1
    return dataset


def create_is_child_feature(dataset):
    dataset["Is_Child"] = dataset["Age"] < 16
    return dataset


def fare_group(fare):
    a = ""
    if fare <= 4:
        a = "very-low"
    elif fare <= 10:
        a = "low"
    elif fare <= 20:
        a = "mid"
    elif fare <= 45:
        a = "high"
    else:
        a = "very-high"
    return a


def create_fare_category_feature(dataset):
    dataset["calculated_fare"] = dataset.Fare / dataset.Family_Size
    dataset["Fare_Category"] = dataset["calculated_fare"].map(fare_group)
    return dataset


def age_group(age):
    a = ""
    if age <= 1:
        a = "infant"
    elif age <= 4:
        a = "toddler"
    elif age <= 13:
        a = "child"
    elif age <= 18:
        a = "teenager"
    elif age <= 35:
        a = "young-adult"
    elif age <= 45:
        a = "adult"
    elif age <= 55:
        a = "middle-aged"
    elif age <= 65:
        a = "senior-citizen"
    else:
        a = "old"
    return a


def create_age_category_feature(dataset):
    dataset["Age_Category"] = dataset["Age"].map(age_group)
    return dataset


def engineer_features(dataset):
    dataset = create_title_feature(dataset)
    dataset = create_family_size_feature(dataset)
    dataset = create_is_alone_feature(dataset)
    dataset = create_is_child_feature(dataset)
    dataset = create_fare_category_feature(dataset)
    dataset = create_age_category_feature(dataset)
    return dataset


# Dataset cleaning
# -----------------------------------------------------------------------
def fill_unknown_ages(dataset):
    title_ages = dict(dataset.groupby("Title")["Age"].median())
    dataset["age_med"] = dataset["Title"].apply(lambda x: title_ages[x])
    dataset["Age"].fillna(dataset["age_med"], inplace=True)
    del dataset["age_med"]
    return dataset


def fill_unknown_fares(dataset):
    class_fares = dict(dataset.groupby("Pclass")["Fare"].median())

    dataset["fare_med"] = dataset["Pclass"].apply(lambda x: class_fares[x])
    dataset["Fare"].fillna(dataset["fare_med"], inplace=True)
    del dataset["fare_med"]
    return dataset


def fill_unknown_embarked(dataset):
    dataset["Embarked"].fillna(method="backfill", inplace=True)
    return dataset


def clean_data(dataset):
    dataset = fill_unknown_ages(dataset)
    dataset = fill_unknown_fares(dataset)
    dataset = fill_unknown_embarked(dataset)
    return dataset


# TODO: heatmaps for correlated features on training set


# Complete Data
# ---------------------------------------------------------------------
def get_datasets():
    setup()
    df = get_concatenated_data()

    # engineer and clean features on df
    # TODO: turn in to class that consumes data and returns instance that includes processed data
    df = engineer_features(df)
    df = clean_data(df)

    train = df[pd.notnull(df["Survived"])]
    test = df[pd.isnull(df["Survived"])]
    return (test, train)


features = [
    "Age",
    "Embarked",
    "Fare",
    "Parch",
    "Pclass",
    "Sex",
    "SibSp",
    # engineered features
    "Title",
    "Family_Size",
    "Family_Category",
    "Is_Alone",
    "Fare_Category",
    "Age_Category",
]
dummy_columns = [
    "Embarked",
    "Pclass",
    "Sex",
    # engineered
    "Title",
    "Is_Alone",
    "Family_Category",
    "Fare_Category",
    "Age_Category",
]


def train_model():
    df_test, df_train = get_datasets()

    training_features = df_train[features]
    X_train = pd.get_dummies(training_features, columns=dummy_columns)
    y_train = df_train["Survived"]

    model = autosklearn.classification.AutoSklearnClassifier(
        ensemble_size=1,
        initial_configurations_via_metalearning=0,
        seed=1,
        n_jobs=-1,
        time_left_for_this_task=60 * 60,
        per_run_time_limit=360,
        resampling_strategy="cv",
        resampling_strategy_arguments={"folds": 3},
        include_preprocessors=["pca"],
        # include_estimators=['random_forest']
    )
    model.fit(X_train, y_train)

    print(model.show_models())
    print(model.sprint_statistics())

    return model


# classifiers = []
# data_preprocessors = []

# for i, (weight, pipeline) in enumerate(model.get_models_with_weights()):
#   for stage_name, component in pipeline.named_steps.items():
#     # print(stage_name)
#     if 'classifier' in stage_name:
#       classifiers.append(component)
#     if 'data_preprocessing' in stage_name:
#       data_preprocessors.append(component)


def create_submission(model):
    (df_test,) = get_datasets()
    testing_features = df_test[features]
    X_test = pd.get_dummies(testing_features, columns=dummy_columns)
    predictions = model.predict(X_test)
    predictions = list(map(lambda x: int(x), predictions))
    test_submission = pd.DataFrame(
        {"PassengerId": df_test.PassengerId, "Survived": predictions}
    )
    test_submission.to_csv("output/tutorial__automl.csv", index=False)
