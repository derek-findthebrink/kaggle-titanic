import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier

from feature_engineering import (
    engineer_features,
    engineered_categorical_features,
    engineered_numerical_features,
)

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


# Complete Data
# ---------------------------------------------------------------------
def extract_training_data(df):
    return df[pd.notnull(df["Survived"])]


def extract_test_data(df):
    return df[pd.isnull(df["Survived"])]


def get_datasets():
    setup()
    df = get_concatenated_data()
    df = engineer_features(df)

    train = extract_training_data(df)
    test = extract_test_data(df)
    return (test, train)


def get_features():
    base_numerical_features = {"Age", "Fare", "Parch", "SibSp"}
    base_categorical_features = {"Pclass", "Embarked", "Sex"}

    numerical_features = base_numerical_features.union(engineered_numerical_features)
    categorical_features = base_categorical_features.union(
        engineered_categorical_features
    )
    return (list(numerical_features), list(categorical_features))


def create_preprocessor():
    numerical_features, categorical_features = get_features()
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    numerical_transformer = Pipeline(
        steps=[
            ("imputer", KNNImputer(n_neighbors=3, weights="distance")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def create_analysis_dataset():
    numerical_features, categorical_features = get_features()
    test, train = get_datasets()
    df = train
    preprocessor = create_preprocessor()
    df_transformed = preprocessor.fit_transform(df)
    # Get feature names after one-hot encoding
    categorical_feature_names = preprocessor.named_transformers_["cat"][
        "onehot"
    ].get_feature_names_out(categorical_features)

    # Combine numerical and categorical feature names
    all_feature_names = numerical_features + list(categorical_feature_names)

    # Convert the numpy array to a DataFrame
    # df_transformed = preprocessor.fit_transform(df)
    df_engineered = pd.DataFrame(df_transformed, columns=all_feature_names)
    df_engineered["Survived"] = df["Survived"]
    return df_engineered


def plot_feature_importance():
    df = create_analysis_dataset()
    X = df.drop("Survived", axis=1)
    y = df["Survived"]
    model = RandomForestClassifier(n_estimators=100, random_state=87)
    model.fit(X, y)
    importances = model.feature_importances_
    feature_names = X.columns

    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x="Importance", y="Feature", data=importance_df)
    plt.title("Feature Importance")
    plt.show()


def lrm():
    return LogisticRegression(max_iter=2000, random_state=87)


def recursive_feature_elimination(X, y, n_features, show_plot=True):
    model = lrm()
    rfe = RFE(model, n_features_to_select=n_features)
    X_rfe = rfe.fit_transform(X, y)
    selected_features = X.columns[rfe.support_]
    print(selected_features)

    if show_plot:
        model_plt = lrm()
        rfe_plt = RFE(model_plt, n_features_to_select=n_features)
        fit = rfe_plt.fit(X, y)
        feature_ranking = pd.Series(fit.ranking_, index=X.columns).sort_values()

        plt.figure(figsize=(10, 8))
        sns.barplot(x=feature_ranking.values, y=feature_ranking.index)
        plt.title("Feature Ranking by RFE")
        plt.show()

    return pd.DataFrame(X_rfe, columns=selected_features)


def plot_correlation_matrixes():
    df_engineered = create_analysis_dataset()

    # Compute the correlation matrix
    correlation_matrix = df_engineered.corr()

    # Get the correlation with the target variable
    correlation_with_survived = correlation_matrix["Survived"].drop("Survived")

    # Display the correlation with the target variable
    print(correlation_with_survived)

    # Bar plot of correlation with Survived
    plt.figure(figsize=(10, 8))
    correlation_with_survived.sort_values(ascending=False).plot(kind="bar")
    plt.title("Correlation of Features with Survived")
    plt.ylabel("Correlation coefficient")
    plt.xlabel("Feature")
    plt.show()

    # Heatmap of correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Matrix")
    plt.show()


class RFETransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_features_to_select=5):
        self.n_features_to_select = n_features_to_select
        self.rfe = None

    def fit(self, X, y=None):
        model = lrm()
        self.rfe = RFE(model, n_features_to_select=self.n_features_to_select)
        self.rfe.fit(X, y)
        return self

    def transform(self, X):
        return self.rfe.transform(X)

    def get_support(self):
        return self.rfe.support_


def create_model():
    xgb_classifier = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    classifier = RandomForestClassifier(n_estimators=100, random_state=87)
    model_1 = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            ("classifier", classifier),
        ]
    )
    model_2 = Pipeline(
        steps=[("preprocessor", create_preprocessor()), ("classifier", xgb_classifier)]
    )
    model_3 = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            ("svc", SVC(probability=True, kernel="rbf")),
        ]
    )
    model_4 = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            ("rfe", RFETransformer(n_features_to_select=20)),
            ("classifier", lrm()),
        ]
    )

    return VotingClassifier(
        estimators=[
            ("rf", model_1),
            ("xgb", model_2),
            ("svc", model_3),
            ("lr", model_4),
        ],
        voting="soft",
        weights=None,
    )


def drop_unused_columns(df):
    return df.drop(["Survived", "Cabin", "Name", "PassengerId", "Ticket"], axis=1)


def get_test_train_split(X, y):
    return train_test_split(X, y, test_size=0.1, random_state=42)


def train_model(model, X, y, show_reports=False):
    if show_reports:
        X_train, X_test, y_train, y_val = get_test_train_split(X, y)
        model.fit(X_train, y_train)
    else:
        model.fit(X, y)

    if show_reports:
        y_pred = model.predict(X_test)
        print(f"\nAccuracy: {accuracy_score(y_val, y_pred)}\n\n")
        print(f"Classification Report:\n{classification_report(y_val, y_pred)}\n\n")
        cm = pd.DataFrame(
            confusion_matrix(y_val, y_pred),
            index=["Actual Positive", "Actual Negative"],
            columns=["Predicted Positive", "Predicted Negative"],
        )
        print(f"Confusion Matrix:\n{cm}")

    return model


from sklearn.inspection import permutation_importance


def plot_permutation_importance():
    df = create_analysis_dataset()
    X = df.drop("Survived", axis=1)
    y = df["Survived"]
    model = RandomForestClassifier(n_estimators=100, random_state=87)
    train_model(model, X, y, show_reports=False)
    results = permutation_importance(
        model, X, y, scoring="accuracy", n_repeats=10, random_state=87
    )
    importance_df = pd.DataFrame(
        {"Feature": X.columns, "Importance": results.importances_mean}
    )
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x="Importance", y="Feature", data=importance_df)
    plt.title("Permutation Feature Importance")
    plt.show()


# plot_permutation_importance()


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
    (df_test, train) = get_datasets()

    # df = drop_unused_columns(df_test)

    predictions = model.predict(df)
    predictions = list(map(lambda x: int(x), predictions))
    test_submission = pd.DataFrame(
        {"PassengerId": df_test.PassengerId, "Survived": predictions}
    )
    test_submission.to_csv("./output/titanic__random_forest.csv", index=False)


df_test, df_train = get_datasets()
X = df_train.drop("Survived", axis=1)
y = df_train["Survived"]

train_model(create_model(), X, y, show_reports=True)

# dfa = create_analysis_dataset()
# dfa_X = dfa.drop("Survived", axis=1)
# dfa_y = dfa["Survived"]
# recursive_feature_elimination(dfa_X, dfa_y, 10)

# create_submission(train_model(create_model(), X, y, show_reports=False))
