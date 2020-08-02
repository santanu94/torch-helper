import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sn

def top_confused(prob, dl, top_k=5, label=None):
    """
    Get top k most confused predictions, with margin of confusion, actual
    label, predicted probability and data.

    Parameters
    ----------
    prob : array-like or tensor
        Probabilities of prediction by model on given dl.
    dl : DataLoader or DataLoaderWrapper
        Train, test, or validation dataloader with valid labels.
    top_k : int, default=5
        The number of most confused outputs to be returned.
    label : int, optional
        If provided, return most confused data for that particular label.
        If None, return most confused across all labels.

    Returns
    -------
    List:
        List of dict with the following keys:
        label - true label for the data.
        pred_probability - predicted probability by model.
        difference - difference between label and pred_probability.
        data - corresponding tensor data.
    """

    # Convert to prob to tensor
    if not torch.is_tensor(prob):
        prob = torch.tensor(prob)

    pred_error_list = []
    start = end = 0
    for xb, yb in dl:
        yb = yb.cpu()
        xb = xb.cpu()

        end += yb.shape[0]
        batch_prob = prob[start:end]

        if label:
            index = torch.where(yb==label)[0]
            yb = yb[index]
            xb = xb[index]
            batch_prob = batch_prob[index]

            if index.nelement() == 0:
                continue

        if batch_prob.dim() == 2 and batch_prob.shape[1] == 1 or batch_prob.dim() == 1:
            pred_label = torch.round(batch_prob)
        if batch_prob.dim() == 2 and batch_prob.shape[1] > 1:# and batch_prob.shape[0] > 0:
            batch_prob, pred_label = torch.max(batch_prob, dim=1)
        diff = torch.abs(batch_prob - yb)

        for i in range(yb.shape[0]):
            pred_error_list.append(
                    {'label': yb[i].item(),
                     'pred_label': pred_label[i].item(),
                     'pred_probability': batch_prob[i].item(),
                     'difference': diff[i].item(),
                     'data': xb[i]
                    })
        start = end

    pred_error_list.sort(reverse=True, key=lambda x: x['difference'])
    return pred_error_list[:top_k]

def plot_confusion_matrix(cm, class_names=None, cmap=plt.cm.Blues):
    h_matrix = sn.heatmap(cm, annot=True, fmt='d', cbar=True, cmap=cmap)
    h_matrix.set_xlabel('Predction')
    h_matrix.set_ylabel('Actual')
    if class_names:
        h_matrix.set_xticklabels(class_names)
        h_matrix.set_yticklabels(class_names)
    plt.show()
