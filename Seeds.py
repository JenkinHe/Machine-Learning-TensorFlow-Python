import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

cols=["area", "perimeter","compactness","length","width","asymmetry","groove","class"]
df = pd.read_csv("seeds_dataset.txt",names=cols,sep="\s+")

print(df.head())

for i in range(len(cols)-1):
    for j in range(i+1,len(cols)-1):
        x_label=cols[i]
        y_label=cols[j]
        # sns.scatterplot(x=x_label,y=y_label,data=df,hue='class')
        # plt.show()

# Clustering

x="perimeter"
y="asymmetry"
X=df[[x,y]].values

kmeans = KMeans(n_clusters=3).fit(X)

clusters=kmeans.labels_

df["class"].values

cluster_df= pd.DataFrame(np.hstack((X,clusters.reshape(-1,1))),columns=[x,y,"class"])

# K means classes



# Higher Dimension

X=df[cols[:-1]].values

kmeans = KMeans(n_clusters=3).fit(X)
cluster_df= pd.DataFrame(np.hstack((X,clusters.reshape(-1,1))),columns=df.columns)

# sns.scatterplot(x=x,y=y,data=cluster_df,hue='class')
# plt.plot()
# plt.show()

# PCA

pcs =PCA(n_components=2)
transformed_x =pcs.fit_transform(X)

print(X.shape)

print(transformed_x.shape)
