import torch
import torch.nn as nn
from ..utils import DataLoaderWrapper

@torch.no_grad()
def predict(model, data):
    """
    Make prediction on data uisng model and return prediction probability, predicted class and label.

    Parameters
    ----------
    model : ModelWrapper or pytorch model
    data: tensor or DataLoaderWrapper
        Pass a transformed tensor to predict on single image.

    Returns
    -------
    tensor
        A tensor of prediction probabilities.
    tensor
        A tensor of prediction class.
    tensor
        A tensor of all labels.
        If labels are not available return tensor of null.
    """

    if type(model) == ModelWrapper:
        model = model.model()

    model.eval()
    if torch.is_tensor(data):
        if len(data.shape) == 3:
            data = torch.unsqueeze(data, 0)
        out = nn.Sigmoid()(model()(data)).cpu()
        return out, torch.round(out).int(), torch.tensor(float('nan'))
    else:
        out_list = pred_list = label_list = None
        for batch in data:
            # Check if batch contains labels
            if len(batch) == 2:
                xb, yb = batch
            else:
                xb = batch[0]
                yb = torch.tensor([float('nan')]*xb.size()[0])

            # Add batch_size dimension if not present
            if len(xb.shape) == 3:
                xb = torch.unsqueeze(xb, 0)

            out = nn.Sigmoid()(model()(xb)).cpu()

            if out_list is None:
                out_list = out
                pred_list = torch.round(out)
                label_list = yb
            else:
                out_list = torch.cat((out_list, out))
                pred_list = torch.cat((pred_list, torch.round(out)))
                label_list = torch.cat((label_list, yb))
        return out_list, pred_list.int(), label_listst.int(), label_list
