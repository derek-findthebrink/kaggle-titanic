import numpy as np
import pandas as pd
import skopt
from sklearn import metrics
from skopt.plots import plot_convergence
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score


# helpers
def get_mae(model, train_X, val_X, train_y, val_y):
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = metrics.mean_absolute_error(val_y, preds_val)
    return mae


training_data = pd.read_csv("../data/train.csv")
test_data = pd.read_csv("../data/test.csv")

# TODO: do proper dimensionality reduction + optimization
# TODO: re-add 'Cabin' here
features = ["Pclass", "Sex", "SibSp", "Parch", "Age", "Embarked"]
dummy_columns = ["Sex", "Embarked"]

# clean up the data before setting up features and dependent
base_features = training_data[features]
base_features = base_features.fillna(base_features.median())

X = pd.get_dummies(base_features, columns=dummy_columns)
y = training_data["Survived"]

print(X.head())


# create base training set
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)


# baseline model
# -----------------------------------------------------------------------------
baseline_model = RandomForestClassifier(
    random_state=1,
    n_estimators=100,
    max_leaf_nodes=10,
)

# baseline_model.fit(train_X, train_y)
# baseline_predicted = baseline_model.predict(val_X)
# baseline_mae = metrics.mean_absolute_error(val_y, baseline_predicted)
baseline_mae = get_mae(baseline_model, train_X, val_X, train_y, val_y)

# Hyperparameter-optimized model
# ------------------------------------------------------------------------------
SPACE = [
    skopt.space.Integer(2, 1000, name="max_leaf_nodes"),
    skopt.space.Integer(2, 200, name="n_estimators"),
    skopt.space.Integer(2, 3000, name="max_depth"),
]

hopt_model = RandomForestClassifier(
    max_depth=5,
    random_state=0,
)


@skopt.utils.use_named_args(SPACE)
def objective(**params):
    hopt_model.set_params(**params)
    cvs = cross_val_score(hopt_model, X, y, cv=5, n_jobs=-1, scoring="accuracy")
    # turning final to a positive number increases MAE (therefore leave it positive)
    final = -np.mean(cvs)
    return final


optimize_results = skopt.gp_minimize(objective, SPACE, n_calls=50, random_state=0)

# test the model with the generated hyperparameters
# baseline model
hopt_model = RandomForestClassifier(
    random_state=0,
    max_leaf_nodes=optimize_results.x[0],
    n_estimators=optimize_results.x[1],
    max_depth=optimize_results.x[2],
)

hopt_mae = get_mae(hopt_model, train_X, val_X, train_y, val_y)


print("")
print(baseline_model)
print({"mae_baseline": baseline_mae})

print("")
print(hopt_model)
print({"mae_hopt": hopt_mae})

# Create competition data
# -------------------------------------------------------------------------------------------
final_model = RandomForestClassifier(
    random_state=0,
    max_leaf_nodes=optimize_results.x[0],
    n_estimators=optimize_results.x[1],
    max_depth=optimize_results.x[2],
)

final_model.fit(X, y)

base_features_test = test_data[features]
base_features_test = base_features_test.fillna(base_features_test.mean())

test_X = pd.get_dummies(base_features_test, columns=dummy_columns)

test_predictions = final_model.predict(test_X)

output = pd.DataFrame(
    {"PassengerId": test_data.PassengerId, "Survived": test_predictions}
)
output.to_csv("output/tutorial__hyperparameter-optimized.csv", index=False)
# print('Doneskis')
