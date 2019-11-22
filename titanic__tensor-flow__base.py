# %%
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# from matplotlib.pyplot import rcParams
# import os

pd.set_option('display.max_row', 1000)
pd.set_option('display.max_columns', 50)


train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

df = pd.concat([train, test], axis=0, sort=True)

# %%
df['Title'] = df['Name'].str.extract('([A-Za-z]+)\.', expand=True)

# TODO: complete this table, lots of values in the df do not have a corresponding map entry
mapping = {
  'Major': 'Mr',
  'Col': 'Mr',
  'Sir': 'Mr',
  'Don': 'Mr',
  'Jonkheer': 'Mr',
  'Capt': 'Mr',
  'Mlle': 'Miss',
  'Mme': 'Mrs',
  'Lady': 'Mrs',
  'Countess': 'Mrs',
  'Dona': 'Mrs',
}

df.replace({ 'Title': mapping }, inplace=True)


# %%

# Computer closer approx. of ages
title_ages = dict(df.groupby('Title')['Age'].median())
df['age_med'] = df['Title'].apply(lambda x: title_ages[x])
df['Age'].fillna(df['age_med'], inplace=True)
del df['age_med']

sns.countplot(x='Title', data=df, hue='Survived')
plt.xticks(rotation=45)
plt.show()

# %%
sns.countplot(x='Sex', data=df, hue='Survived')

# %%
sns.countplot(x='Title', data=df)

# %%
class_fares = dict(df.groupby('Pclass')['Fare'].median())

df['fare_med'] = df['Pclass'].apply(lambda x: class_fares[x])
df['Fare'].fillna(df['fare_med'], inplace=True)
del df['fare_med']

# %%
df['Embarked'].fillna(method='backfill', inplace=True)

# %%
df['Family_Size'] = df['Parch'] + df['SibSp']

# %% [markdown]

# At this point, all data processing is complete. Training sets and test sets can be broken
# up by selecting based on whether or not the survived feature is set

# %%
train = df[pd.notnull(df['Survived'])]
test = df[pd.isnull(df['Survived'])]

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV
# from keras.wrappers.scikit_learn import KerasClassifier
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Dropout

# from numpy.random import seed

# %%
df.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

continuous = ['Age', 'Fare', 'Parch', 'Pclass', 'SibSp', 'Family_Size']

scaler = StandardScaler()

for var in continuous:
  df[var] = df[var].astype('float64')
  df[var] = scaler.fit_transform(df[var].values.reshape(-1, 1))

df.head()

# %%
X_train = df[pd.notnull(df['Survived'])].drop(['Survived'], axis=1)
y_train = df[pd.notnull(df['Survived'])]['Survived']
X_test = df[pd.isnull(df['Survived'])].drop(['Survived'], axis=1)

# %%
import tensorflow as tf

Sex = tf.feature_column.categorical_column_with_vocabulary_list('Sex', ['female', 'male'])
Embarked = tf.feature_column.categorical_column_with_vocabulary_list('Embarked', ['S', 'C', 'Q'])
Title = tf.feature_column.categorical_column_with_vocabulary_list('Title', ['Mr', 'Mrs', 'Miss', 'Master', 'Rev', 'Dr', 'Ms'])
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

# %%
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

# %%

# final_predictions = model.predict()
# output = pd.DataFrame({ 'PassengerId': test.PassengerId, 'Survived': final_predictions })

# # %%
# output.to_csv('output/tutorial__tensorflow.csv', index=False)