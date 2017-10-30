import os
import numpy as np
import sklearn

from sklearn.model_selection import train_test_split


default_dirs = [\
    "data/transport1-10-1000-2-100-2-4-B/",
    "data/transport-like-D12/",
    "data/transport-like-D22/",
    "data/park-4-6-B/",
    "data/park-5-7-B/",
    "data/park-5-8-B/",
    "data/park-6-8-B/",
    "data/elevators-12-3/",
    "data/no-mystery-like-M22/"
]

#default_columns = [5,8,10,11,12,13] # v006: Hamming distance and FF derived features
#default_columns = [5,8,10,11,12,13,14,15,16] # v007: v006 + single
#default_columns = [5,8,10,11,12,13] + range(17,17+9) # v008: v006 + pair
#default_columns = [5,8,10,11,12,13,14,15,16] + range(17,17+9) #v009: v006 + single + pair
default_columns = [5,8,10,11,12,13,14,15,16,17,18] # FF DI (AAAI-18)

data_limits = [7000, 7000, 7000, 5000, 5000, 5000, 5000, 20000, 20000]

def read(dirs=default_dirs, columns=default_columns, test_size=0.1):

    feature_files = []
    label_files = []
    for d in dirs:
        f_dir = os.path.join(d, 'features')
        #for f in sorted(os.listdir(f_dir)):
        #    feature_files.append(os.path.join(f_dir, f))
        feature_files.append([os.path.join(f_dir, f) for f in sorted(os.listdir(f_dir))])
        l_dir = os.path.join(d, 'labels')
        #for l in sorted(os.listdir(l_dir)):
        #    label_files.append(os.path.join(l_dir, l))
        label_files.append([os.path.join(l_dir, l) for l in sorted(os.listdir(l_dir))])

    features = None
    labels = None

    skip_goal = False

    # It got messy as a result of adding new functionality and should be rewritten. In particular, the feature and label files shoud be read in lockstep to enable correctness checks.
    for (fl, limit) in zip(feature_files, data_limits):
        features_dir = None
        for f in fl:
            features_file = np.loadtxt(f)
            if skip_goal:
                features_file = features_file[:-1]
            if features_dir is not None:
                features_dir = np.append(features_dir, features_file, 0)
            else:
                features_dir = features_file
            if len(features_dir) > limit:
                break
        if columns:
            features_dir = features_dir[:, columns]
        if features is not None:
            print features.shape, features_dir.shape
            features = np.append(features, features_dir, 0)
        else:
            features = features_dir

    for (fl, limit) in zip(label_files, data_limits):
        labels_dir = None
        for f in fl:
            labels_file = np.loadtxt(f)
            if skip_goal:
                labels_file = labels_file[:-1]
            if labels_dir is not None and labels_dir.shape != ():
                if labels_file.shape != ():
                    print labels_dir.shape, labels_file.shape
                    labels_dir = np.append(labels_dir, labels_file, 0)
            else:
                labels_dir = labels_file
            if len(labels_dir) > limit:
                break
        if labels is not None:
            print labels.shape, labels_dir.shape
            labels = np.append(labels, labels_dir, 0)
        else:
            labels = labels_dir

    #if columns:
    #    features = features[:, columns]
    
    print features.shape, labels.shape
    
    return train_test_split(features, labels, train_size=(1-test_size), random_state=None)
    


