import numpy as np
from typing import List, Tuple
import torch

def compute_acc_without_pad(
    y_pred, 
    y_test
) -> Tuple[float, List[float], List[float]]: 

    """
    Computes accuracy and returns accuracy, as well as predictions and labels without padding.

    :param y_pred: Tensor of predictions for N examples, size (N, N_classes)
    :param y_test: Tensor of labels for N examples, size (N, 1)
    :return: Accuracy, list of predictions, list of labels
    """
    
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    pred_vals = [ float(np.argmax(item)) for item in y_pred_tag.detach() ]
    labels = [ float(item) for item in y_test.detach() ]

    # Remove frames which have label -100 (this comes from padding)
    keep_indexes = []
    for idx, label in enumerate(labels):
        if label != float(-100):
            keep_indexes.append(idx)

    pred_vals = [pred_vals[i] for i in keep_indexes]
    labels = [labels[i] for i in keep_indexes]

    correct_results_sum = sum(a == b for a, b in zip(pred_vals, labels))
    acc = correct_results_sum / len(labels)
    acc = np.round(acc * 100)

    return acc, pred_vals, labels
