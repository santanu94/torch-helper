import torch
from sklearn.metrics import f1_score

def get_accuracy(y_true, y_pred):
    # return torch.tensor(torch.sum(output_label_similarity).item() / len(output_label_similarity))
    return torch.sum(y_true == y_pred).item() / len(y_true)

def get_f1_score(y_true, y_pred, average):
    print(y_pred.device)
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().detach()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().detach()
    if average in ['binary', 'micro', 'macro', 'weighted', 'samples']:
        return f1_score(y_true, y_pred, average=average)
    else:
        print("f1_score value unidentified. Accepted values are 'binary', 'micro', 'macro', 'weighted', 'samples'")
    return 0
