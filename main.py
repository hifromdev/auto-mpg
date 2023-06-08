import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

path_to_file = "auto-mpg.data"
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(path_to_file, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)

dataset = raw_dataset.copy()

# .isna() in pandas checks for null/NA/NaN values and .sum() counts the occurences of null/... values.
# The statement below indicates that there are six data entries where horsepower is not provided.
print(dataset.isna().sum())

# This removes the rows with NA values.
dataset = dataset.dropna()
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
print(dataset.tail())

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

#   Uncomment to see dataplots.
# sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
# plt.show()

# This shows us how there is a different range of values in every label.
print(train_dataset.describe().transpose())

# Since we're predicting MPG, we're splitting MPG 'label' from the rest of the features.
# Splitting allows us to isolate the data that we will predict from the data that we will use to predict.
train_features = train_dataset.copy()
test_features = test_dataset.copy()
train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

# 
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))
print(normalizer.mean.numpy())

# first = np.array(train_features[:1])

# with np.printoptions(precision=2, suppress=True):
#   print('First example:', first)
#   print()
#   print('Normalized:', normalizer(first).numpy())

test_results = {}

# 
linear_model = tf.keras.Sequential([
  normalizer,
  layers.Dense(units=1)
])

linear_model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
  loss='mean_absolute_error'
)

history = linear_model.fit(
  train_features,
  train_labels,
  epochs=100,
  # Suppress logging.
  verbose=0,
  # Calculate validation results on 20% of the training data.
  validation_split = 0.2
)

plot_loss(history)
plt.show()

test_results['linear_model'] = linear_model.evaluate(
  test_features, 
  test_labels, 
  verbose=0
)

print(test_results['linear_model'])