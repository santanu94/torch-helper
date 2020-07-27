import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def top_confused(prob, dl, top_k=5, class_no=None):
    pred_error_list = []
    start = end = 0
    for xb, yb in dl:
        yb = yb.cpu()
        xb = xb.cpu()

        if class_no and int(class_no) != int(yb):
            continue

        end += yb.shape[0]
        diff = torch.abs(prob[start:end] - yb)

        for i in range(yb.shape[0]):
            pred_error_list.append(
                    {'label': yb[i].item(),
                     'prediction': prob[start:end][i].item(),
                     'difference': diff[i].item(),
                     'data': xb[i]
                    })
        start = end

    pred_error_list.sort(reverse=True, key=lambda x: x['difference'])
    return pred_error_list[:top_k]
