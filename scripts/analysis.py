from hippospharm.clustering.learning import clustering


def main():
    clustering('data/features/features.csv', metadata_csv='data/metadata/participants.tsv', num_clusters=2,
               labels='age', method='aecm-kmeans')

if __name__ == '__main__':
    main()