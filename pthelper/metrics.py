import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sn
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

def get_classifier_accuracy(y_hat, y_true, classifier):
    """
    Get accuracy for binary/multi-class classifier.

    Parameters
    ----------
    y_hat : tensor
        Model output logit
    y_true : tensor
        True label tensor
    classifier : 'binary' or 'multi'
        Applies torch.round() for binary and torch.argmax() over nn.Softmax(dim=1) for multi
    
    Returns
    -------
    int :
        Accuracy of y_hat against y_true
    """
    if classifier == 'binary':
        y_pred = torch.round(nn.Sigmoid()(y_hat))
    elif classifier == 'multi':
        y_pred = torch.argmax(nn.Softmax(dim=1)(y_hat), dim=1)
    else:
        raise Exception("Unknown classifier: Only ['binay', 'multi'] supported")
    return torch.sum(y_true == y_pred).item() / len(y_true)

def get_accuracy(y_pred, y_true):
    """
    Get accuracy of predicted labels against actual labels.

    Parameters
    ----------
    y_hat : tensor, numpy array or list
        Final predicted labels that can be compared against the labels y_true, must be same shape as y_true
    y_true : tensor, numpy array or list
        True label array
    
    Returns
    -------
    int :
        Accuracy of y_pred against y_true
    """
    if not torch.is_tensor(y_pred):
        y_pred = torch.IntTensor(y_pred)
    if not torch.is_tensor(y_true):
        y_true = torch.IntTensor(y_true)
    assert y_pred.shape == y_true.shape
    return torch.sum(y_true == y_pred).item() / len(y_true)

def get_f1_score(y_pred, y_true, average):
    """
    Get f1 score of predicted labels against actual labels. Uses Scikit-Learn API.

    Parameters
    ----------
    y_hat : tensor, numpy array or list
        Final predicted labels that can be compared against the labels y_true, must be same shape as y_true
    y_true : tensor, numpy array or list
        True labels
    
    Returns
    -------
    int :
        Accuracy of y_hat against y_true
    """
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().detach()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().detach()
    if average in ['binary', 'micro', 'macro', 'weighted', 'samples']:
        return f1_score(y_true, y_pred, average=average)
    else:
        print("f1_score value unidentified. Accepted values are 'binary', 'micro', 'macro', 'weighted', 'samples'")
    return 0

def top_confused(x, y_hat, y_true, model_type, top_k=5, label=None):
    """
    Get top k most confused predictions.

    Parameters
    ----------
    x : tensor, numpy array or list
        Model input data
    y_hat : tensor
        Model output after applying final activation layer
    y_true : tensor, numpy array or list
        True labels
    model_type : int or String
        'regression' or 0 : return top confused based on euclidean distance of y_hat from y_true
        'binary-class' or 1 : return top confused based on difference in true label (1 or 0) and model output probability
        'multi-class' or 2 : return top confused based on difference in model output probability for true class and 1
    top_k : int, default=5
        The number of most confused outputs to be returned
    label : int, optional
        If provided, return most confused data for that particular label or regression value
        If None, return most confused across all labels

    Returns
    -------
    List :
        List of dict with the following keys if model_type parameter is 0 or regression:
            true_value - true value for the data
            pred_value - predicted value by the model
            difference - difference between true_value and pred_value.
            data - corresponding data.
        
        And, following keys if model_type parameter is 1, 2, binary-class or multi-class:
            true_label - true label for the data.
            pred_label - predicted label by model
            pred_probability - predicted probability by model.
            difference - difference between true_label and pred_probability.
            data - corresponding tensor data.
    """
    if label:
        y_hat = y_hat.detach().clone()
        y_true = y_true.detach().clone()
        x = x.detach().clone()

        y_true_eq_label = y_true == label
        y_hat = y_hat[y_true_eq_label]
        x = x[y_true_eq_label]
        y_true = y_true[y_true_eq_label]

    max_prob_diff_idx = np.empty((0, 2))
    pred_error_list = []
    for idx, y_hat_val in enumerate(tqdm(y_hat, leave=False)):
        if model_type in [0, 'regression', 1, 'binary-class']:
            diff = torch.abs(y_true[idx] - y_hat_val.squeeze().item())
        elif model_type in [2, 'multi-class']:
            diff = 1 - y_hat_val[y_true[idx]].item()
        else:
            raise Exception("Unknown model_type value. Should be one of [0, 'regression', 1, 'binary-class', 2, 'multi-class']")
        
        if max_prob_diff_idx.shape[0] < top_k:
            max_prob_diff_idx = np.append(max_prob_diff_idx, [[diff, idx]], axis=0)
            continue

        if max_prob_diff_idx[max_prob_diff_idx[:, 0] < diff].shape[0] > 0:
            max_prob_diff_idx[max_prob_diff_idx[:, 0].min()] = [diff, idx]
    
    assert max_prob_diff_idx.shape[0] == top_k
    for diff, idx in max_prob_diff_idx[np.argsort(max_prob_diff_idx[:, 1])[::-1]]:
        if model_type in [0, 'regression']:
            pred_error_list.append(
                {'true_value': y_true[idx],
                 'pred_value': y_hat[idx],
                 'difference': diff,
                 'data': x[idx]
                })
        elif model_type in [2, 'multi-class']:
            prob = y_hat[idx]
            pred_error_list.append(
                {'true_label': y_true[idx],
                 'pred_label': torch.round(prob),
                 'pred_probability': prob,
                 'difference': diff,
                 'data': x[idx]
                })
        elif model_type in [2, 'multi-class']:
            probs = y_hat[idx]
            pred_error_list.append(
                {'true_label': y_true[idx],
                 'pred_label': torch.argmax(probs),
                 'pred_probability': probs,
                 'difference': diff,
                 'data': x[idx]
                })

    return pred_error_list

def plot_confusion_matrix(cm, class_names=None, cmap=plt.cm.Blues):
    """Plot Confusion Matrix heatmap with Seaborn API."""
    h_matrix = sn.heatmap(cm, annot=True, fmt='d', cbar=True, cmap=cmap)
    h_matrix.set_xlabel('Predction')
    h_matrix.set_ylabel('Actual')
    if class_names:
        h_matrix.set_xticklabels(class_names)
        h_matrix.set_yticklabels(class_names)
    plt.show()
