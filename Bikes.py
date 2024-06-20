import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import copy
import seaborn as sns
import tensorflow as tf
from sklearn.linear_model import LinearRegression


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
    if  x_label is None:
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

temp_reg = LinearRegression()
temp_reg.fit(x_train_temp,y_train_temp)

# print(temp_reg.score(x_test_temp,y_test_temp))

# plt.scatter(x_train_temp,y_train_temp,label="Data",color="blue")
# x=tf.linspace(-20,40,100)
# plt.plot(x,temp_reg.predict(np.array(x).reshape(-1,1)),label="Fit",color="red",linewidth=3)
# plt.legend()
# plt.title("Bikes vs temp")
# plt.ylabel("number of bikes")
# plt.xlabel("Temp")
# plt.show()


# Multiple linear regression
_, x_train_all,y_train_all=get_xy(train,"bike_count",x_label=df.columns[1:])
_, x_val_all,y_val_all=get_xy(val,"bike_count",x_label=df.columns[1:])
_, x_test_all,y_test_all=get_xy(test,"bike_count",x_label=df.columns[1:])

all_reg = LinearRegression()
all_reg.fit(x_train_all,y_train_all)

print(all_reg.score(x_test_all,y_test_all))

# Regression with a neural net

def plot_history(history):

    plt.plot(history.history['loss'],label='loss')
    plt.plot(history.history['val_loss'],label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.show()


# temp_normalizer=tf.keras.layers.Normalization(input_shape=(1,),axis=None)
# temp_normalizer.adapt((x_train_temp).reshape(-1))

# temp_nn_model =tf.keras.Sequential([
#     temp_normalizer,
#     tf.keras.layers.Dense(1)
# ])

# temp_nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),loss='mean_squared_error')

# history= temp_nn_model.fit(
#     x_train_temp.reshape(-1), y_train_temp,
#     # verbose=0,
#     epochs=1000,
#     validation_data=(x_val_temp,y_val_temp)
# )

# plot_history(history)

# Neural net
# temp_normalizer=tf.keras.layers.Normalization(input_shape=(1,),axis=None)
# temp_normalizer.adapt((x_train_temp).reshape(-1))
# nn_model =tf.keras.Sequential([
#     temp_normalizer,
#     tf.keras.layers.Dense(32,activation='relu'),
#     tf.keras.layers.Dense(32,activation='relu'),
#     tf.keras.layers.Dense(1,activation='relu')
# ])

# nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss='mean_squared_error')

# history= nn_model.fit(
#     x_train_temp,y_train_temp,
#     epochs=100,
#     validation_data=(x_val_temp,y_val_temp)
# )


# plot_history(history)

# all data neural net

all_normalizer=tf.keras.layers.Normalization(input_shape=(6,),axis=-1)
all_normalizer.adapt(x_train_all)
nn_model =tf.keras.Sequential([
    all_normalizer,
    tf.keras.layers.Dense(32,activation='relu'),
    tf.keras.layers.Dense(32,activation='relu'),
    tf.keras.layers.Dense(1)
])

nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss='mean_squared_error')

history= nn_model.fit(
    x_train_all,y_train_all,
    validation_data=(x_val_all,y_val_all),
    epochs=100,
)

# calculate the mse for both linear reg and nn
y_pred_lr=all_reg.predict(x_test_all)
y_pred_nn=nn_model.predict(x_test_all)

def MSE(y_pred,y_real):
    return(np.square(y_pred-y_real)).mean()

MSE(y_pred_lr,y_test_all)
MSE(y_pred_nn,y_test_all)

axis = plt.axes(aspect="equal")
plt.scatter(y_test_all,y_pred_lr,label="Lin Reg Preds")
plt.xlabel("True Values")
plt.ylabel("Predictions")
lims=[0,2500]
plt.xlim(lims)
plt.ylim(lims)
_=plt.plot(lims,lims,c="red")
plt.show()