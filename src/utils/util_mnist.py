import matplotlib.pyplot as plt
import numpy as np
import random
import os

def plot_digits(train_dataset, general_reports_dir):
    fig, axs = plt.subplots(5, 5, figsize=(8,8))
    for ax in axs.flatten():
        # random.choice allows to randomly sample from a list-like object (basically anything that can be accessed with an index, like our dataset)
        img, label = random.choice(train_dataset)
        ax.imshow(np.array(img[0]), cmap='gist_gray')
        ax.set_title('Label: %d' % label)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(os.path.join(general_reports_dir, "mnist.png"), dpi=400, format='png')
    # plt.show()

def split_MNIST(dataset, iid_digits):

    classes = []
    indices = dataset.targets == iid_digits[0]
    classes.append(dataset.classes[iid_digits[0]])
    if len(iid_digits) > 1:
        for digit in iid_digits[1:]:
            idx = dataset.targets == digit
            indices = indices + idx
            classes.append(dataset.classes[digit])

    dataset.targets = dataset.targets[indices]
    dataset.data = dataset.data[indices]
    dataset.classes = classes

    return dataset

def create_binary_dataset(dataset, iid_digits):

    classes = []
    indices = dataset.targets == iid_digits[0]
    classes.append(dataset.classes[iid_digits[0]])
    if len(iid_digits) > 1:
        for digit in iid_digits[1:]:
            idx = dataset.targets == digit
            indices = indices + idx
            classes.append(dataset.classes[digit])

    dataset.targets = dataset.targets[indices]
    dataset.data = dataset.data[indices]
    dataset.classes = classes

    return dataset