# Feature engineering
# ----------------------------------------------------------------------
engineered_categorical_features = set()
engineered_numerical_features = set()


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
    engineered_categorical_features.add("Title")
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
    engineered_categorical_features.add("Family_Category")
    engineered_numerical_features.add("Family_Size")
    return dataset


def create_is_alone_feature(dataset):
    dataset["Is_Alone"] = dataset["Family_Size"] == 1
    engineered_categorical_features.add("Is_Alone")
    return dataset


def create_is_child_feature(dataset):
    dataset["Is_Child"] = dataset["Age"] < 16
    engineered_categorical_features.add("Is_Child")
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
    dataset["Calculated_Fare"] = round(dataset.Fare / dataset.Family_Size, 2)
    dataset["Fare_Category"] = dataset["Calculated_Fare"].map(fare_group)
    engineered_categorical_features.add("Fare_Category")
    del dataset["Calculated_Fare"]
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
        a = "unknown"
    return a


def create_age_category_feature(dataset):
    dataset["Age_Category"] = dataset["Age"].map(age_group)
    engineered_categorical_features.add("Age_Category")
    return dataset


def engineer_features(dataset):
    dataset = create_title_feature(dataset)
    dataset = create_family_size_feature(dataset)
    dataset = create_is_alone_feature(dataset)
    dataset = create_is_child_feature(dataset)
    dataset = create_fare_category_feature(dataset)
    dataset = create_age_category_feature(dataset)
    return dataset
