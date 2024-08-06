import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

# read csv file
dataset = pd.read_csv('/data/bot_detection_data.csv')

# Split independent and dependant variables
X = dataset.iloc[:, 3:7]
y = dataset.iloc[:, 7]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Perform feature scaling on the dataset
# Not necessary with decision tree algorithm
sc = StandardScaler()
X_train.iloc[:, :3] = sc.fit_transform(X_train.iloc[:, :3])
X_test.iloc[:, :3] = sc.transform(X_test.iloc[:, :3])

# Train the model
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)
