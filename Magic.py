import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import tensorflow as tf

cols =["fLength","fWidth","fSize","fConc","fConc1","fAsym","fM3Long","fM3Trans","fAlpha","fDist","class"]
df = pd.read_csv("magic04.data",names=cols)

# print(df.head())

df["class"] =(df["class"]=="g").astype(int)

# for label in cols[:-1]:
#     plt.hist(df[df["class"]==1][label],color='blue',label='gamma',alpha=0.7,density=True)
#     plt.hist(df[df["class"]==0][label],color='red',label='hadron',alpha=0.7,density=True)
#     plt.title(label)
#     plt.ylabel("Probability")
#     plt.xlabel(label)
#     plt.legend()
#     plt.show()

# Train, validation , test datasets

train,valid,test= np.split(df.sample(frac=1),[int(0.6*len(df)),int(0.8*len(df))])

# scaling data

def scale_dataset(dataframe,oversample=False):
    x =dataframe[dataframe.columns[:-1]].values
    y=dataframe[dataframe.columns[-1]].values

    scaler=StandardScaler()

    x= scaler.fit_transform(x)

    if oversample:
        ros =RandomOverSampler()
        x,y=ros.fit_resample(x,y)

    data=np.hstack((x,np.reshape(y,(-1,1))))

    return data,x,y

train, x_train, y_train = scale_dataset(train,oversample=True)
valid, x_valid, y_valid = scale_dataset(valid,oversample=False)
test, x_test, y_test = scale_dataset(test,oversample=False)

# K nearest neighbours


knn_model=KNeighborsClassifier(n_neighbors=5)
knn_model.fit(x_train,y_train)

y_pred =knn_model.predict(x_test)

# print(classification_report(y_test,y_pred))
# Naive bayes model
nb_model= GaussianNB()
nb_model=nb_model.fit(x_train,y_train)

y_pred =nb_model.predict(x_test)

# print(classification_report(y_test,y_pred))

# Logistic regression

lg_model=LogisticRegression()
lg_model= lg_model.fit(x_train, y_train)
y_pred =lg_model.predict(x_test)

# print(classification_report(y_test,y_pred))

# Support vector machine //not great with outliers

svm_model= SVC()
svm_model = svm_model.fit(x_train,y_train)

y_pred =svm_model.predict(x_test)

print(classification_report(y_test,y_pred))

# Neural network

def plot_loss(history):
    plt.plot(history.history['loss'],label='loss')
    plt.plot(history.history['val_loss'],label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Binary crossentropy')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_accuracy(history):
    plt.plot(history.history['accuracy'],label='accuracy')
    plt.plot(history.history['val_accuracy'],label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

nn_model = tf.keras.Sequential([
    tf.keras.layers.Dense(32,activation='relu',input_shape=(10,)),
    tf.keras.layers.Dense(32,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

nn_model.compile(optimizer=tf.keras.optimizers.Adam(0.001),loss='binary_crossentropy',metrics=['accuracy'])

