# Import library

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Read credit card data file
#df = pd.read_csv("processed.cleveland.data", sep=",")
df = pd.read_csv("creditcard.csv", sep=",")

# class distribution and plotting

print("No Fraud '0': {} % of the dataset".format(round(df['Class'].value_counts(1)[0] * 100,2)))
print("Fraud '1': {} '% of the dataset \n".format(round(df['Class'].value_counts(1)[1] * 100,2)))

colors = ["#0101DF", "#DF0101"]

sns.countplot('Class', data=df, palette=colors)
plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)
