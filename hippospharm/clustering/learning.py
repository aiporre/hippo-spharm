import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from hippospharm.clustering.embedding import Embedding


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

def clustering(feat_csv, method='PCA-Kmeans', num_clusters=2, labels=None, metadata_csv=None , dims=2):
    # read csv
    feats = pd.read_csv(feat_csv)
    if metadata_csv is not None:
        feats = load_metadata(features=feats, metadata_csv=metadata_csv)
        # filter right or left
        #side = 'left'
        #feats = feats.loc[feats['side'] == side]

        # get labels
        if labels == 'age':
            y = feats['age'].values
            y  = (y>30).astype('float32')
        elif labels == 'sex':
            y = feats['sex'].values
            # transform to 0 and 1 from M and F
            y = (y == 'M').astype('float32')
        elif labels == 'side':
            y = feats['side'].values
            # transform to 0 and 1 from L and R
            y = (y == 'left').astype('float32')
        else:
            y = feats['side'].values
    print(feats.head())
    # get features
    feat_cols = [c for c in feats.columns if 'feat' in c]
    X = feats[feat_cols].values

    # create embedding
    emb = Embedding(method=method, dims=dims, num_clusters=
            num_clusters)
    z, clusters = emb.fit(X, y=y)
    # plot
    if labels is None:
        # only plots clusters
        plt.figure()
        for i in range(num_clusters):
            plt.scatter(z[clusters ==i,0], z[clusters ==i,1], label='cluster {}'.format(i))
        plt.legend()
        plt.show()
    else:
        fig, ax = plt.subplots(1,2, figsize=(10,5))
        for i in range(num_clusters):
            ax[0].scatter(z[clusters ==i,0], z[clusters ==i,1], label='cluster {}'.format(i))
        ax[0].legend()
        # plot color by side
        if  labels == 'side':
            sides = feats['side'].values
            for i in range(len(sides)):
                if sides[i] == 'left':
                    ax[1].scatter(z[i,0], z[i,1], c='b')
                elif sides[i] == 'right':
                    ax[1].scatter(z[i,0], z[i,1], c='r')
        elif labels == 'age':
            ages = feats['age'].values
            # if age is less that 30 years old, color is blue and if age is more than 30 years old, color is red
            for i in range(len(ages)):
                if ages[i] < 30:
                    ax[1].scatter(z[i,0], z[i,1], c='b')
                else:
                    ax[1].scatter(z[i, 0], z[i, 1], c='r')
        elif labels == 'sex':
            sexs = feats['sex'].values
            # if sex is M blue if sex is F red
            for i in range(len(sexs)-1):
                if sexs[i] == 'M':
                    ax[1].scatter(z[i,0], z[i,1], c='b')
                else:
                    ax[1].scatter(z[i, 0], z[i, 1], c='r')
        # make a legend related to the second label
        ax[1].legend()
        plt.show()
    # save
    feats[f'cluster_{method}'] = clusters
    feats.to_csv(feat_csv, index=False)