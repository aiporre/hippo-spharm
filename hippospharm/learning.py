import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class Embedding:
    def __init__(self, method, dims=2, **kwargs):
        self.method = method
        self.kwargs = kwargs
        self.dims = dims

    def fit(self, X):
        if self.method == 'PCA':
            self.model = PCA(n_components = self.dims, **self.kwargs)
            x_emb = self.model.fit_transform(X)
        elif self.method == 'TSNE':
            from sklearn.manifold import TSNE
            self.model = TSNE(n_components = self.dims, **self.kwargs)
            x_emb = self.model.fit_transform(X)

        return x_emb


def clustering(feat_csv, num_clusters=2):
    # read csv
    df = pd.read_csv(feat_csv)
    # get features
    feat_cols = [c for c in df.columns if 'feat' in c]
    X = df[feat_cols].values
    subs = df['subject'].values
    # normalize
    X_std = StandardScaler().fit_transform(X)
    # PCA
    pca = PCA(n_components=2)
    pca.fit(X_std)
    X_pca = pca.transform(X_std)
    # KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X_pca)
    # plot
    fig, ax = plt.subplots()
    for i in range(num_clusters):
        ax.scatter(X_pca[kmeans.labels_==i,0], X_pca[kmeans.labels_==i,1], label='cluster {}'.format(i))
    ax.legend()
    plt.show()
    # save
    df['cluster_pca'] = kmeans.labels_
    df.to_csv(feat_csv, index=False)