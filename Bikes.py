import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import copy
import seaborn as sns
import tensorflow as tf
from sklearn.linear_model import LogisticRegression


# Linear Regression
dataset_cols = ["bike_count","hour","temp","humidity","wind","visibility","dew_pt_temp","radiation","rain","snow","functional"]

df= pd.read_csv("SeoulBikeData.csv").drop(["Date","Holiday","Seasons"],axis=1)
df.columns=dataset_cols
df["functional"]=(df["functional"]!="Yes").astype(int)
df = df[df["hour"]==12]
df= df.drop(["hour"],axis=1)


# for label in df.columns[1:]:
#     plt.scatter(df[label],df["bike_count"])
#     plt.title(label)
#     plt.ylabel("Bike count at noon")
#     plt.xlabel(label)
#     plt.show()


df=df.drop(["wind","visibility","functional"],axis=1)
print(df.head())
