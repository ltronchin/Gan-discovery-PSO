from sklearn.neighbors import KNeighborsClassifier

import os
import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np


def load_classifiers(model_dir):  # Pretrained classifiers
    with open(os.path.join(model_dir, "classifiers"), 'rb') as open_file:
        classifiers = pickle.load(open_file)
    return classifiers


def train_classifier_battery(encoded_samples_train):
    # Training phase
    print("Training phase")
    val_size = int(len(encoded_samples_train)* 0.2) # lets reserve 20% of our data for validation
    label_tr = np.unique(encoded_samples_train["label"])
    clf_pts = {}
    for label in tqdm(label_tr):
        # Create N new dataset with boolean label where N is equal to the number of class
        x = encoded_samples_train.iloc[:, :-1].to_numpy()
        y = np.array(encoded_samples_train["label"] == label, dtype="uint8")
        # Extract train data
        x_train = x[:-val_size]
        y_train = y[:-val_size]

        clf =  KNeighborsClassifier(n_neighbors=3)
        clf_pts[label] = clf.fit(x_train, y_train)

    return clf_pts

def compute_posterior(encoding, classifiers, assign_label=None):
    print("Compute Posterior from classifiers battery")
    encoding = encoding.to_numpy()
    p_yx = np.zeros(shape=(encoding.shape[0], len(classifiers)))
    for idx_enc, enc in enumerate(encoding):  # for idx_enc, enc in enumerate(tqdm(encoding)):  # over images
        for idx, idx_iid_label in enumerate(classifiers):  # over classes in distribution!
            clf = classifiers[idx_iid_label]
            p_yx[idx_enc, idx] = clf.predict_proba(np.expand_dims(enc, axis=0))[0][1]
    p_yx = pd.DataFrame(p_yx)

    if assign_label is not None:
        # Compute softmax along rows
        p_yx_softmax = softmax(p_yx)
        # Compute argmax along rows to extract the classification column (patient classification)
        y_encoding = np.argmax(p_yx_softmax, axis=1)
        # Create Dataframe with synthetic encoding along with synthetic label
        encoding = pd.DataFrame(
            data={'var_0': encoding[:, 0], 'var_1': encoding[:, 1], 'label': y_encoding})
        return p_yx, encoding
    else:
        return p_yx

def softmax(x):
    max_row = np.max(x, axis=1, keepdims=True)  # returns max of each row and keeps same dims
    e_x = np.exp(x - max_row)  # subtracts each row with its max value
    sum_row = np.sum(e_x, axis=1, keepdims=True)  # returns sum of each row and keeps same dims
    f_x = e_x / sum_row
    return f_x