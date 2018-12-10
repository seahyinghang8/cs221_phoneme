import os
import cPickle as pkl
import yaml

import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

PROCESS_DIR = os.path.dirname(os.path.realpath(__file__))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '',#format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def plot_phoneme_cm(predictions, phn_index_map):
    """
        Plot a confusion matrix after evaluation

        Parameters
        ----------
            - predictions: dict: a dict with keys y_true and y_pred, both mapping to a list
            - phn_index_map: dict: an index mapping the phoneme to a label
        Returns
        -------
            None

    """
    cm = confusion_matrix(predictions['y_true'], predictions['y_pred'], labels=None)

    classes = range(int(cm.shape[0]))

    for phn, idx in phn_index_map.items():
        if idx < len(classes):
            if isinstance(classes[idx], int):
                classes[idx] = phn
            else:
                classes[idx] += ', {}'.format(phn)

    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cm, classes=classes,
                          normalize=True,
                          title='Confusion matrix')
    plt.show()

if __name__ == '__main__':
    pass