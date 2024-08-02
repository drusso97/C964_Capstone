import torch

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('/data/bot_detection_data.csv')
x = dataset.iloc[:, 2:5]
y = dataset.iloc[:, :7]