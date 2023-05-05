from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


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
            z = self.model.transform(X_std)
            # KMeans
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=0).fit(z)
            clusters = kmeans.labels_
        elif self.method.lower == 'tsne-kmeans':
            from sklearn.manifold import TSNE
            self.model = TSNE(n_components = self.dims, **self.kwargs)
            z = self.model.fit_transform(X)
            # KMeans
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=0).fit(z)
            clusters = kmeans.labels_
        elif self.method.lower() == 'aecm':
            from hippospharm.clustering.embedding_AECM import AECM
            self.model = AECM(n_components = self.dims, **self.kwargs)
            error =  self.model.fit(X)
            z = self.model.predict(X)
            # clusting
            clusters = self.model.predict_km
        return z, clusters
