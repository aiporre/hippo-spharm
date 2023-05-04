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
def load_metadata(features, metadata_csv):
    # read data and ajust to shape to the features
    metadata = pd.read_csv(metadata_csv)
    features['age'] = np.zeros(features.shape[0])
    features['sex'] = np.zeros(features.shape[0])
    for participant_id in metadata['participant_id']:
        # copy age and sex in the corresponding features to the participant
        # matching columns subjects and participant_id in features and metadata respectively.
        features.loc[features['subject'] == participant_id, 'age'] = metadata.loc[metadata['participant_id'] == participant_id, 'age'].values[0]
        features.loc[features['subject'] == participant_id, 'sex'] = metadata.loc[metadata['participant_id'] == participant_id, 'sex'].values[0]
    return features




def clustering(feat_csv, num_clusters=2, metadata_csv=None, second_label=None):
    # read csv
    feats = pd.read_csv(feat_csv)
    if metadata_csv is not None:
        feats = load_metadata(features=feats, metadata_csv=metadata_csv)
    print(feats.head())
    # get features
    feat_cols = [c for c in feats.columns if 'feat' in c]
    X = feats[feat_cols].values
    # normalize
    X_std = StandardScaler().fit_transform(X)
    # PCA
    pca = PCA(n_components=2)
    pca.fit(X_std)
    X_pca = pca.transform(X_std)
    # KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X_pca)
    # plot
    if second_label is None:
        # only plots clusters
        plt.figure()
        for i in range(num_clusters):
            plt.scatter(X_pca[kmeans.labels_==i,0], X_pca[kmeans.labels_==i,1], label='cluster {}'.format(i))
        plt.legend()
        plt.show()
    else:
        fig, ax = plt.subplots(1,2, figsize=(10,5))
        for i in range(num_clusters):
            ax[0].scatter(X_pca[kmeans.labels_==i,0], X_pca[kmeans.labels_==i,1], label='cluster {}'.format(i))
        ax[0].legend()
        # plot color by side
        if second_label == 'side':
            sides = feats['side'].values
            for i in range(len(sides)):
                if sides[i] == 'left':
                    ax[1].scatter(X_pca[i,0], X_pca[i,1], c='b')
                elif sides[i] == 'right':
                    ax[1].scatter(X_pca[i,0], X_pca[i,1], c='r')
        elif second_label == 'age':
            ages = feats['age'].values
            # if age is less that 30 years old, color is blue and if age is more than 30 years old, color is red
            for i in range(len(ages)):
                if ages[i] < 30:
                    ax[1].scatter(X_pca[i,0], X_pca[i,1], c='b')
                else:
                    ax[1].scatter(X_pca[i,0], X_pca[i,1], c='r')
        # make a legend related to the second label
        ax[1].legend()
        plt.show()
    # save
    feats['cluster_pca'] = kmeans.labels_
    feats.to_csv(feat_csv, index=False)