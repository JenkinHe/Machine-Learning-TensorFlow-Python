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

# Train/valid/test dataset

train,val,test=np.split(df.sample(frac=1), [int(0.6*len(df)),int(0.8*len(df))])

def get_xy(dataframe,y_label, x_label=None):
    dataframe=copy.deepcopy(dataframe)
    if not x_label:
        x= dataframe[[c for c in dataframe.columns if c!=y_label]].values
    else:
        if len(x_label)==1:
            x= dataframe[x_label[0]].values.reshape(-1,1)
        else:
            x=dataframe[x_label].values
    
    y=dataframe[y_label].values.reshape(-1,1)
    data= np.hstack((x,y))

    return data,x,y

_, x_train_temp,y_train_temp=get_xy(train,"bike_count",x_label=["temp"])
_, x_val_temp,y_val_temp=get_xy(val,"bike_count",x_label=["temp"])
_, x_test_temp,y_test_temp=get_xy(test,"bike_count",x_label=["temp"])


