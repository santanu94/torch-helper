import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def top_confused(prob, dl, top_k=5, label=None):
    if torch.is_tensor(prob):
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


        diff = torch.abs(batch_prob - yb)

        for i in range(yb.shape[0]):
            pred_error_list.append(
                    {'label': yb[i].item(),
                     'prediction': batch_prob[i].item(),
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
    h_matrix.set_xticklabels(class_names)
    h_matrix.set_yticklabels(class_names)
    plt.show()
