# %%
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# TODO: Figure out proper way to treat dataset reassignment in python, without using class instance variables

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



# Setup
# ----------------------------------------------------------------------
def setup():
  pd.set_option('display.max_row', 1000)
  pd.set_option('display.max_columns', 50)

def get_training_data():
  return pd.read_csv('./data/train.csv')

def get_test_data():
  return pd.read_csv('./data/test.csv')

def get_concatenated_data():
  train = get_training_data()
  test = get_test_data()
  return pd.concat([train, test], axis=0, sort=True)



# Feature engineering
# ----------------------------------------------------------------------
def create_title_feature(dataset):
  # Targets are: Mr, Mrs, Master, Ms, Miss
  # Must remove entries where key === value because replace function complains
  #  - alternative is to use map, however using map resulted in a few NaN values
  #    persisting in the frame for this feature
  mapping = {
    'Capt'        : 'Mr',
    'Col'         : 'Mr',
    'Countess'    : 'Mrs',
    # problematic, best guess is there were more men as doctors back then
    # => !should be computed from dataset!
    'Dr'          : 'Mr',
    'Don'         : 'Mr',
    'Dona'        : 'Mrs',
    'Jonkheer'    : 'Mr',
    'Lady'        : 'Mrs',
    'Major'       : 'Mr',
    # 'Master'      : 'Master',
    # 'Miss'        : 'Miss',
    'Mlle'        : 'Miss',
    'Mme'         : 'Mrs',
    # 'Mr'          : 'Mr',
    # 'Mrs'         : 'Mrs',
    # 'Ms'          : 'Ms',
    'Rev'         : 'Mr',
    'Sir'         : 'Mr',
  }

  # extract Title feature (intermediate) from name feature
  dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.', expand=True)
  dataset.replace({ 'Title': mapping }, inplace=True)
  return dataset

def create_family_size_feature(dataset):
  dataset['Family_Size'] = dataset['Parch'] + dataset['SibSp']
  return dataset



def engineer_features(dataset):
  dataset = create_title_feature(dataset)
  dataset = create_family_size_feature(dataset)
  return dataset



# Dataset cleaning
# -----------------------------------------------------------------------
def fill_unknown_ages(dataset):
  title_ages = dict(dataset.groupby('Title')['Age'].median())
  dataset['age_med'] = dataset['Title'].apply(lambda x: title_ages[x])
  dataset['Age'].fillna(dataset['age_med'], inplace=True)
  del dataset['age_med']
  return dataset

def fill_unknown_fares(dataset):
  class_fares = dict(dataset.groupby('Pclass')['Fare'].median())

  dataset['fare_med'] = dataset['Pclass'].apply(lambda x: class_fares[x])
  dataset['Fare'].fillna(dataset['fare_med'], inplace=True)
  del dataset['fare_med']
  return dataset

def fill_unknown_embarked(dataset):
  dataset['Embarked'].fillna(method='backfill', inplace=True)
  return dataset



def clean_data(dataset):
  dataset = fill_unknown_ages(dataset)
  dataset = fill_unknown_fares(dataset)
  dataset = fill_unknown_embarked(dataset)
  return dataset

# TODO: heatmaps for correlated features on training set



# Utilities
# ---------------------------------------------------------------------
def find_columns_with_null(dataset):
  null_columns = dataset.columns[dataset.isnull().any()]
  return dataset[null_columns].isnull().sum()


# Complete Data
# ---------------------------------------------------------------------
def get_datasets():
  setup()
  df = get_concatenated_data()

  # %%
  # engineer and clean features on df
  # TODO: turn in to class that consumes data and returns instance that includes processed data
  df = engineer_features(df)
  df = clean_data(df)

  train = df[pd.notnull(df['Survived'])]
  test = df[pd.isnull(df['Survived'])]
  return (test, train)












# sns.countplot(x='Title', data=df, hue='Survived')
# plt.xticks(rotation=45)
# plt.show()

# sns.countplot(x='Sex', data=df, hue='Survived')

# sns.countplot(x='Title', data=df)



# %%
# Setup model functions
def clean_data_for_model(df):
  from sklearn.preprocessing import StandardScaler

  dataset = df.copy().drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1)

  continuous = ['Age', 'Fare', 'Parch', 'Pclass', 'SibSp', 'Family_Size']
  scaler = StandardScaler()
  for var in continuous:
    dataset[var] = dataset[var].astype('float64')
    dataset[var] = scaler.fit_transform(dataset[var].values.reshape(-1, 1))
  
  return dataset


def train_model(dataset):
  from sklearn.model_selection import train_test_split
  # from sklearn.model_selection import GridSearchCV
  # from keras.wrappers.scikit_learn import KerasClassifier
  # from keras.models import Sequential
  # from keras.layers import Dense, Activation, Dropout

  # from numpy.random import seed

  dataset = clean_data_for_model(dataset)

  X_train = dataset[pd.notnull(dataset['Survived'])].drop(['Survived'], axis=1)
  y_train = dataset[pd.notnull(dataset['Survived'])]['Survived']

  import tensorflow as tf

  Sex = tf.feature_column.categorical_column_with_vocabulary_list('Sex', ['female', 'male'])
  Embarked = tf.feature_column.categorical_column_with_vocabulary_list('Embarked', ['S', 'C', 'Q'])
  Title = tf.feature_column.categorical_column_with_vocabulary_list('Title', ['Mr', 'Mrs', 'Master', 'Ms', 'Miss'])
  Age = tf.feature_column.numeric_column('Age')
  Fare = tf.feature_column.numeric_column('Fare')
  Parch = tf.feature_column.numeric_column('Parch')
  Pclass = tf.feature_column.numeric_column('Pclass')
  SibSp = tf.feature_column.numeric_column('SibSp')
  Family_Size = tf.feature_column.numeric_column('Family_Size')

  feat_cols = [Sex, Embarked, Title, Age, Fare, Parch, Pclass, SibSp, Family_Size]

  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

  input_func = tf.estimator.inputs.pandas_input_fn(
    x=X_train,
    y=y_train,
    batch_size=100,
    num_epochs=None,
    shuffle=True
  )

  model = tf.estimator.LinearClassifier(feature_columns=feat_cols)
  model.train(input_fn=input_func, max_steps=10000)

  pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_val, batch_size=len(X_val), shuffle=False)
  predictions = list(model.predict(input_fn=pred_fn))

  final_predictions = []

  # %%
  for pred in predictions:
    final_predictions.append(pred['class_ids'][0])

  from sklearn.metrics import classification_report
  print(classification_report(y_val, final_predictions))

  return model

# %%

df_test, df_train = get_datasets()
model = train_model(df_train)



# %%

# final_predictions = model.predict()
# output = pd.DataFrame({ 'PassengerId': test.PassengerId, 'Survived': final_predictions })
df_test_clean = clean_data_for_model(df_test)

# # %%
# output.to_csv('output/tutorial__tensorflow.csv', index=False)


# %%

import tensorflow as tf

pred_fn = tf.estimator.inputs.pandas_input_fn(x=df_test_clean, batch_size=len(df_test_clean), shuffle=False)
predictions = list(model.predict(input_fn=pred_fn))
final_predictions = []
for pred in predictions:
  final_predictions.append(pred['class_ids'][0])


# %%
test_submission = pd.DataFrame({'PassengerID': df_test.PassengerId, 'Survived': final_predictions })
test_submission.to_csv('output/tutorial__tensorflow.csv', index=False)
# %%
