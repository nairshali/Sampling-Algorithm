# Import library

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Read credit card data file
#df = pd.read_csv("processed.cleveland.data", sep=",")
df = pd.read_csv("creditcard.csv", sep=",")
