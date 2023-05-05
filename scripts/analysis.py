from hippospharm.clustering.learning import clustering


def main():
    clustering('data/features/features.csv', metadata_csv='data/metadata/participants.tsv', num_clusters=5,
               labels='age')

if __name__ == '__main__':
    main()