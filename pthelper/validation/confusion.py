import torch

def top_confused(prob, dl, top_k=5, class_no=None):
    if torch.is_tensor(prob):
        prob = prob.tolist()
    pred_error_list = []
    start = end = None
    for xb, yb in dl:
        if class_no and int(class_no) != int(yb):
            continue

        end = yb.shape[0]
        diff = abs(prob[start:end] - yb)
        pred_error_list.append(
                {'label': yb[i],
                 'prediction': prob[i],
                 'difference': diff[i],
                 'data': xb[i]
                } for i in range(yb.shape[0]))
        start = start + end

    pred_error_list.sort(reverse=True, key=lambda x: x.difference)
    return pred_error_list[:top_k]
