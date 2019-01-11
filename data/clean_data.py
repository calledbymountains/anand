import pandas as pd
import os
import glob
from bidict import bidict

drop_indices = [0, 1, 2, 4]

output_indices = [16,17,18,19]

SUMMARY_MAPPING = {
    'Clear': 0,
    'Drizzle' : 1,
    'Flurries' : 2,
    'Foggy' : 3,
    'Light Rain' : 4,
    'Light Snow' : 5,
    'Mostly Cloudy' : 6,
    'Overcast' : 7,
    'Partly Cloudy' : 8,
    'Rain' : 9,
    'Snow' : 10
}

def clean_file(filename):
    if not os.path.isfile(filename):
        raise ValueError('The file {} does not exist.'.format(filename))

    df = pd.read_csv(filename, delimiter=',')
    header_names = df.columns.values.tolist()
    df.dropna(axis=0, inplace=True)
    df_labels = df[[header_names[x] for x in output_indices]]

    df_features = df.drop(labels=[header_names[x] for x in output_indices], axis=1, inplace=False)
    df_features.drop(labels=[header_names[x] for x in drop_indices], axis=1, inplace=True)


    return df_features, df_labels

def getallfiles(filepath):
    if not os.path.isdir(filepath):
        raise ValueError('The path {} does not exist.'.format(filepath))

    files = glob.glob(os.path.join(filepath, '*.csv'))

    for ind, file in enumerate(files):
        if ind == 0:
            df_features, df_labels = clean_file(file)
        else:
            temp_features, temp_labels = clean_file(file)
            df_features = pd.concat([df_features, temp_features], axis=0)
            df_labels = pd.concat([df_labels, temp_labels], axis=0)

        print('Read {}'.format(file))

    return df_features, df_labels


if __name__ == "__main__":
    filepath = os.getcwd()
    df_features, df_labels = getallfiles(filepath)
    df_features['summary'] = df_features.summary.astype('category')
    df_features['summary'] = df_features['summary'].apply(
        lambda x: SUMMARY_MAPPING[x] + 1)
    print(df_features['summary'])