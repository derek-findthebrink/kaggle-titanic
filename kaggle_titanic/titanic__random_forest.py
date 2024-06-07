# import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    StackingClassifier,
    GradientBoostingClassifier,
)
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier

import seaborn as sns
import matplotlib.pyplot as plt

from feature_engineering import (
    engineer_features,
    engineered_categorical_features,
    engineered_numerical_features,
)

from analysis import (
    plot_correlation_matrixes,
    plot_feature_importance,
    plot_permutation_importance,
    print_model_accuracy_report,
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


def drop_unused_columns(df):
    return df.drop(["Cabin", "Name", "PassengerId", "Ticket"], axis=1)


base_numerical_features = {"Age", "Fare", "Parch", "SibSp"}
base_categorical_features = {"Pclass", "Embarked", "Sex"}


def get_datasets(drop=True):
    setup()
    df = get_concatenated_data()
    df = engineer_features(df)
    if drop:
        df = drop_unused_columns(df)

    train = extract_training_data(df)
    test = extract_test_data(df)
    return (test, train)


def get_features():
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


class RFETransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_features_to_select=5):
        self.n_features_to_select = n_features_to_select
        self.rfe = None

    def fit(self, X, y=None):
        model = lrc()
        self.rfe = RFE(model, n_features_to_select=self.n_features_to_select)
        self.rfe.fit(X, y)
        return self

    def transform(self, X):
        return self.rfe.transform(X)

    def get_support(self):
        return self.rfe.support_


def lrc():
    return LogisticRegression(max_iter=2000, random_state=87)


def lrc_pipeline():
    return Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            ("rfe", RFETransformer(n_features_to_select=20)),
            ("classifier", lrc()),
        ]
    )


def rfc():
    return RandomForestClassifier(n_estimators=100, random_state=87)


def rfc_pipeline():
    return Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            ("classifier", rfc()),
        ]
    )


def xgbc():
    return XGBClassifier(
        use_label_encoder=False, eval_metric="logloss", random_state=87
    )


def xgbc_pipeline():
    return Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            ("classifier", xgbc()),
        ]
    )


def svc():
    return SVC(probability=True, kernel="rbf", random_state=87)


def svc_pipeline():
    return Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            ("svc", svc()),
        ]
    )


def gbc():
    return GradientBoostingClassifier(n_estimators=100, random_state=87)


def knn():
    return KNeighborsClassifier(n_neighbors=5)


def knn_pipeline():
    return Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            ("classifier", knn()),
        ]
    )


def get_base_models():
    return [
        ("rf", rfc_pipeline()),
        ("xgb", xgbc_pipeline()),
        ("svc", svc_pipeline()),
        ("lr", lrc_pipeline()),
    ]


def voting_pipeline():
    return VotingClassifier(
        estimators=get_base_models(),
        voting="soft",
        weights=None,
    )


def stacking_pipeline():
    return StackingClassifier(
        estimators=get_base_models(),
        final_estimator=lrc(),
        cv=5,
        stack_method="predict_proba",
    )


def recursive_feature_elimination(X, y, n_features, show_plot=True):
    model = lrc()
    rfe = RFE(model, n_features_to_select=n_features)
    X_rfe = rfe.fit_transform(X, y)
    selected_features = X.columns[rfe.support_]
    print(selected_features)

    if show_plot:
        model_plt = lrc()
        rfe_plt = RFE(model_plt, n_features_to_select=n_features)
        fit = rfe_plt.fit(X, y)
        feature_ranking = pd.Series(fit.ranking_, index=X.columns).sort_values()

        plt.figure(figsize=(10, 8))
        sns.barplot(x=feature_ranking.values, y=feature_ranking.index)
        plt.title("Feature Ranking by RFE")
        plt.show()

    return pd.DataFrame(X_rfe, columns=selected_features)


def create_analysis_dataset():
    numerical_features, categorical_features = get_features()
    _, train = get_datasets()
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


# classifiers = []
# data_preprocessors = []

# for i, (weight, pipeline) in enumerate(model.get_models_with_weights()):
#   for stage_name, component in pipeline.named_steps.items():
#     # print(stage_name)
#     if 'classifier' in stage_name:
#       classifiers.append(component)
#     if 'data_preprocessing' in stage_name:
#       data_preprocessors.append(component)


def train_model(model, X, y):
    model.fit(X, y)

    return model


def create_submission(model):
    df_test, _ = get_datasets(drop=False)

    predictions = model.predict(df_test)
    predictions = list(map(lambda x: int(x), predictions))
    test_submission = pd.DataFrame(
        {"PassengerId": df_test.PassengerId, "Survived": predictions}
    )
    test_submission.to_csv("./output/titanic__random_forest.csv", index=False)


df_test, df_train = get_datasets()
X = df_train.drop("Survived", axis=1)
y = df_train["Survived"]

# train_model(voting_pipeline(), X, y)
# train_model(stacking_pipeline(), X, y)

# dfa = create_analysis_dataset()
# dfa_X = dfa.drop("Survived", axis=1)
# dfa_y = dfa["Survived"]
# recursive_feature_elimination(dfa_X, dfa_y, 10)

# plot_correlation_matrixes(create_analysis_dataset())
# plot_feature_importance(create_analysis_dataset())
# plot_permutation_importance(create_analysis_dataset())
# print_model_accuracy_report(rfc_pipeline(), X, y, label="Random Forest Classifier")
# print_model_accuracy_report(xgbc_pipeline(), X, y, label="XGBoost Classifier")
# print_model_accuracy_report(svc_pipeline(), X, y, label="Support Vector Classifier")
# print_model_accuracy_report(
#     lrc_pipeline(), X, y, label="Logistic Regression Classifier"
# )
print_model_accuracy_report(voting_pipeline(), X, y, label="Voting Classifier")
# print_model_accuracy_report(stacking_pipeline(), X, y, label="Stacking Classifier")

create_submission(train_model(voting_pipeline(), X, y))
