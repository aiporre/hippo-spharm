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
        # side = 'left'
        # feats = feats.loc[feats['side'] == side]

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
        elif labels == 'all':
            print('plotting all labels')
            y = None
        else:
            y = feats['side'].values
    print(feats.head())
    # get features
    feat_cols = [c for c in feats.columns if 'feat' in c]
    X = feats[feat_cols].values[:, :500]
    # y = feats['age'].values
    # y = (y>30).astype('float32')
    # mask_1 = y == 1
    # mask_2 = y == 0
    # center_1 = np.mean(X[mask_1], axis=0)
    # center_2 = np.mean(X[mask_2], axis=0)
    # # move random
    # center_1 = center_1 + 10*np.ones(center_1.shape)
    # center_2 = center_2 #+ np.random.normal(0, 0.1, center_2.shape)
    # X1 = X[mask_1] - center_1
    # X2 = X[mask_2] - center_2
    # X = np.concatenate([X1, X2], axis=0)
    # y = np.concatenate([y[mask_1], y[mask_2]], axis=0)
    # feats['age'] = y*50 + 25
    # y=None

    # create embedding
    emb = Embedding(method=method, dims=dims, num_clusters=
            num_clusters, epochs=1000)
    z, clusters = emb.fit(X, y=y)
    # plot
    if labels is None:
        # only plots clusters
        plt.figure()
        for i in range(num_clusters):
            plt.scatter(z[clusters ==i,0], z[clusters ==i,1], label='cluster {}'.format(i))
        plt.legend()
        plt.show()
    elif labels == 'all':
        fig, ax = plt.subplots(1,3, figsize=(15,5), layout='constrained')
        cmap = plt.get_cmap('jet')
        norm = plt.Normalize(clusters.min(), clusters.max())
        for c in np.unique(clusters):
            mask = clusters == c
            # ax[0].scatter(z[mask,0], z[mask,1], c=cmap(norm(c)), label='cluster {}'.format(c))
            ax[0].scatter(z[mask, 0], z[mask, 1], label='{}'.format(c))
        ax[0].legend(loc='upper right',bbox_to_anchor=(-0.15, 1.05),
                     fancybox=True, shadow=True, ncol=1)

        # ax[0].scatter(z[:,0], z[:,1], c=clusters)
        y = feats['age'].values
        s = ax[1].scatter(z[:,0], z[:,1], c=y)
        plt.colorbar(s, ax=ax[1], location='bottom')


        # for age in np.unique(y):
        #     mask = y == age
        #     ax[1].scatter(z[mask,0], z[mask,1], c=cmap(norm(age)), label='age {}'.format(age))
        # ax[1].legend()
        # ax[1].set_title('age')
        # plot sex
        y = feats['sex']
        y = (y == 'M').astype('float32')
        mask_male = y == 1
        ax[2].scatter(z[mask_male,0], z[mask_male,1], c='b', label='M')
        mask_female = y == 0
        ax[2].scatter(z[mask_female,0], z[mask_female,1], c='r', label='F')
        ax[2].legend()
        # plt.colorbar(ax=ax)
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