import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_decision_forests as tfdf

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


from feature_engineering import engineer_features


# Complete Data
# ---------------------------------------------------------------------
def extract_training_data(df):
    return df[pd.notnull(df["Survived"])]


def extract_test_data(df):
    return df[pd.isnull(df["Survived"])]


def drop_unused_columns(df):
    return df.drop(
        [
            "Cabin",
            "Name",
            "PassengerId",
            "Ticket",
            "Family_Size",
            "Family_Category",
            "Is_Alone",
            "Is_Child",
            "Fare_Category",
            "Age_Category",
        ],
        axis=1,
    )


def get_datasets(drop=True):
    df = get_concatenated_data()
    df = engineer_features(df)
    df = convert_boolean_columns(df)
    if drop:
        df = drop_unused_columns(df)

    train = extract_training_data(df)
    test = extract_test_data(df)
    return (test, train)


def convert_boolean_columns(df):
    bool_cols = df.select_dtypes(include=[bool]).columns
    for col in bool_cols:
        df[col] = df[col].astype(int)
    return df


(test, train) = get_datasets(drop=False)
# import ipdb

# ipdb.set_trace()

input_features = list(train.columns)
input_features.remove("Ticket")
input_features.remove("PassengerId")
input_features.remove("Survived")


def tokenize_names(features, labels=None):
    features["Name"] = tf.strings.split(features["Name"])
    return features, labels


train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train, label="Survived").map(
    tokenize_names
)
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test, label="Survived").map(
    tokenize_names
)

model = tfdf.keras.GradientBoostedTreesModel(
    verbose=0,
    features=[tfdf.keras.FeatureUsage(name=n) for n in input_features],
    exclude_non_specified_features=True,
    random_seed=87,
    min_examples=1,
    categorical_algorithm="RANDOM",
    # max_depth=4,
    shrinkage=0.05,
    # num_candidate_attributes_ratio=0.2,
    split_axis="SPARSE_OBLIQUE",
    sparse_oblique_normalization="MIN_MAX",
    sparse_oblique_num_projections_exponent=2.0,
    num_trees=2000,
)
# model.fit(train_ds)

# self_eval = model.make_inspector().evaluation()
# print(f"(BASE) Accuracy: {self_eval.accuracy}, Loss: {self_eval.loss}")
# model.summary()

tuner = tfdf.tuner.RandomSearch(num_trials=2000)
tuner.choice("min_examples", [2, 5, 7, 10])
tuner.choice("categorical_algorithm", ["CART", "RANDOM"])

local_search_space = tuner.choice("growing_strategy", ["LOCAL"])
local_search_space.choice("max_depth", [3, 4, 5, 6, 8])

global_search_space = tuner.choice(
    "growing_strategy", ["BEST_FIRST_GLOBAL"], merge=True
)
global_search_space.choice("max_num_nodes", [16, 32, 64, 128, 256])

tuner.choice("shrinkage", [0.1, 0.05, 0.10, 0.15])
tuner.choice("num_candidate_attributes_ratio", [0.2, 0.5, 0.9, 1.0])

tuner.choice("split_axis", ["AXIS_ALIGNED"])
oblique_space = tuner.choice("split_axis", ["SPARSE_OBLIQUE"], merge=True)
oblique_space.choice(
    "sparse_oblique_normalization", ["NONE", "STANDARD_DEVIATION", "MIN_MAX"]
)
oblique_space.choice("sparse_oblique_weights", ["BINARY", "CONTINUOUS"])
oblique_space.choice("sparse_oblique_num_projections_exponent", [1.0, 1.5])

# Tune the model. Notice the `tuner=tuner`.
tuned_model = tfdf.keras.GradientBoostedTreesModel(tuner=tuner)
tuned_model.fit(train_ds, verbose=1)

tuned_self_evaluation = tuned_model.make_inspector().evaluation()
print(
    f"(TUNED) Accuracy: {tuned_self_evaluation.accuracy} Loss:{tuned_self_evaluation.loss}"
)


def make_submission(model, df, ds, threshold=0.5):
    predictions = model.predict(ds, verbose=0)[:, 0]

    submission = pd.DataFrame(
        {
            "PassengerId": df["PassengerId"],
            "Survived": (predictions >= threshold).astype(int),
        }
    )
    submission["Survived"] = submission["Survived"].astype(int)
    submission.to_csv("output/titanic__tensorflow_2.csv", index=False)


# make_submission(model, test, test_ds)
make_submission(tuned_model, test, test_ds)
