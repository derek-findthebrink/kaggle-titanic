# import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


def plot_correlation_matrixes(df):
    # Compute the correlation matrix
    correlation_matrix = df.corr()

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


def plot_feature_importance(df):
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


def plot_permutation_importance(df):
    X = df.drop("Survived", axis=1)
    y = df["Survived"]
    model = RandomForestClassifier(n_estimators=100, random_state=87)
    model.fit(X, y)
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


def print_model_accuracy_report(model, X, y, label=None):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=87
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if label:
        print(f"\n{label}\n-----------------------------------------------")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}\n")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}\n")
    cm = pd.DataFrame(
        confusion_matrix(y_test, y_pred),
        columns=["Predicted Died", "Predicted Survived"],
        index=["Actual Died", "Actual Survived"],
    )
    print(f"Confusion Matrix:\n{cm}\n\n")
