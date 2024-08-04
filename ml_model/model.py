import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# read csv file
dataset = pd.read_csv('/data/bot_detection_data.csv')

# Split independent and dependant variables
x = dataset.iloc[:, 3:7]
y = dataset.iloc[:, 7]

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
