import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cols =["fLength","fWidth","fSize","fConc","fConc1","fAsym","fM3Long","fM3Trans","fAlpha","fDist","class"]
df = pd.read_csv("magic04.data",names=cols)

print(df.head())

df["class"] =(df["class"]=="g").astype(int)

for label in cols[:-1]:
    plt.hist(df[df["class"]==1][label],color='blue',label='gamma',alpha=0.7,density=True)
    plt.hist(df[df["class"]==0][label],color='red',label='hadron',alpha=0.7,density=True)
    plt.title(label)
    plt.ylabel("Probability")
    plt.xlabel(label)
    plt.legend()
    plt.show()

# Train validation , test datasets

train,valid,test= np.split(df.sample(frac=1),[int(0.6*len(df)),int(0.8*len(df))])

