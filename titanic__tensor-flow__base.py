# %%
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
import os

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
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

from numpy.random import seed

# %%
df.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

continuous = ['Age', 'Fare', 'Parch', 'Pclass', 'SibSp', 'Family_Size']

# %%

scaler = StandardScaler()

for var in continuous:
  df[var] = df[var].astype('float64')
  df[var] = scaler.fit_transform(df[var].values.reshape(-1, 1))

df.head()