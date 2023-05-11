from hippospharm.clustering.learning import clustering
import argparse

def main(features, metadata, num_clusters, labels, method, dims):
    clustering(features, metadata_csv=metadata, num_clusters=num_clusters,labels=labels, method=method, dims=dims)

if __name__ == '__main__':
    # make a arguments
    parser = argparse.ArgumentParser(description='Clustering hippocampus surfaces')
    parser.add_argument('--features', type=str, help='path to features csv file')
    parser.add_argument('--metadata', type=str, help='path to metadata csv file')
    parser.add_argument('--num_clusters', type=int, help='number of clusters', default=2)
    parser.add_argument('--labels', type=str, help='column name in metadata to use as labels', default='age')
    parser.add_argument('--method', type=str, help='clustering method', default='AECM-KMeans')
    parser.add_argument('--dims', type=int, help='number of dimensions for embedding', default=2)
    args = parser.parse_args()
    main(args.features, args.metadata, args.num_clusters, args.labels, args.method, args.dims)