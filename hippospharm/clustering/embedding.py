from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from sklearn.manifold import TSNE
from hippospharm.clustering.embedding_AECM import EmbeddingAECM


class Embedding:
    def __init__(self, method, dims=2, num_clusters=2, **kwargs):
        self.method = method
        self.kwargs = kwargs
        self.dims = dims
        self.num_clusters = num_clusters

    def fit(self, X, y=None):
        if self.method.lower() == 'pca-kmeans':
            # normalize
            X_std = StandardScaler().fit_transform(X)
            # PCA
            self.model = PCA(n_components=self.dims)
            self.model.fit(X_std)
            z = self.model.transform(X)
            # KMeans
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=0).fit(z)
            clusters = kmeans.labels_
            if self.dims>2:
                dim_model = TSNE(n_components = 2)
                z = dim_model.fit_transform(z)
        elif self.method.lower() == 'tsne-kmeans':
            self.model = TSNE(n_components = self.dims)
            z = self.model.fit_transform(X)
            # KMeans
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=0).fit(z)
            clusters = kmeans.labels_
            if self.dims>2:
                dim_model = TSNE(n_components = 2)
                z = dim_model.fit_transform(z)
        elif self.method.lower() == 'kmeans':
            z = X
            # KMeans
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=0).fit(z)
            clusters = kmeans.labels_
            if self.dims > 2:
                dim_model = TSNE(n_components=2)
                z = dim_model.fit_transform(z)
        elif self.method.lower() == 'aecm':
            from hippospharm.clustering.embedding_AECM import AECM
            X = X.astype('float32')
            self.model = EmbeddingAECM(n_clusters=self.num_clusters, dims=self.dims,
                                       X=X, y=y, **self.kwargs)
            error =  self.model.fit(X, y)
            z = self.model.embedding(X)
            # clustering
            clusters = self.model.predict(X)
            # dimensioality reduction
            if self.dims>2:
                dim_model = TSNE(n_components = 2)
                z = dim_model.fit_transform(z)
        elif self.method.lower() == 'aecm-kmeans':
            from hippospharm.clustering.embedding_AECM import AECM
            # X = StandardScaler().fit_transform(X)
            X = X.astype('float32')
            self.model = EmbeddingAECM(n_clusters=self.num_clusters, dims=self.dims,
                                       X=X, y=y, **self.kwargs)
            error = self.model.fit(X, y)
            z = self.model.embedding(X)
            # clustering
            clusters = self.model.predict_kmeans(X)
            # dimensioality reduction
            if self.dims>2:
                dim_model = TSNE(n_components = 2)
                z = dim_model.fit_transform(z)
        else:
            raise ValueError('Method {} not implemented'.format(self.method))
        return z, clusters
