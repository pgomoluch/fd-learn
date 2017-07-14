import os
import numpy as np
import sklearn

from sklearn.model_selection import train_test_split


default_dirs = [\
    #'data/park-4-6',
    #'data/park-5-7',
    #'data/park-5-8',
    #'data/park-6-8',
    'data/transport1-10-1000-2-100-2-4',
    'data/transport1-10-1000-2-100-2-4-B',
    'data/transport1-10-1000-2-100-2-4-C',
    #'data/wood-p2-rw10000',
]

#default_columns = [5,8,10,11,12,13] # v006: Hamming distance and FF derived features
#default_columns = [5,8,10,11,12,13,14,15,16] # v007: v006 + single
#default_columns = [5,8,10,11,12,13] + range(17,17+9) # v008: v006 + pair
default_columns = [5,8,10,11,12,13,14,15,16] + range(17,17+9) #v009: v006 + single + pair

def read(dirs=default_dirs, columns=default_columns, test_size=0.1):

    feature_files = []
    label_files = []
    for d in dirs:
        f_dir = os.path.join(d, 'features')
        for f in sorted(os.listdir(f_dir)):
            feature_files.append(os.path.join(f_dir, f))
        l_dir = os.path.join(d, 'labels')
        for l in sorted(os.listdir(l_dir)):
            label_files.append(os.path.join(l_dir, l))

    features = None
    labels = None

    skip_goal = False

    for f in feature_files:
        features_chunk = np.loadtxt(f)
        if skip_goal:
            features_chunk = features_chunk[:-1]
        if features is not None:
            features = np.append(features, features_chunk, 0)
        else:
            features = features_chunk

    for f in label_files:
        labels_chunk = np.loadtxt(f)
        if skip_goal:
            labels_chunk = labels_chunk[:-1]
        if labels is not None:
            labels = np.append(labels, labels_chunk, 0)
        else:
            labels = labels_chunk

    if columns:
        features = features[:, columns]

    return train_test_split(features, labels, train_size=(1-test_size), random_state=None)
    


