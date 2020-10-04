import torch
import torch.nn as nn
from ..training import ModelWrapper
from ..utils import to_device

@torch.no_grad()
def predict(model, data, pred_func=None, output_selection_func=None):
    """
    Make prediction on data uisng model and return prediction probability, predicted class and label.

    Parameters
    ----------
    model : ModelWrapper or pytorch model
    data: Tensor or DataLoaderWrapper
        Pass a transformed tensor to predict on single image.
    pred_func : Pytorch layer or function reference, optional
        If model outputs logit then pass a function reference or a pytorch
        layer to get prediction or calculate model accuracy.
        For regression output, keep None.
        Not required if model is a ModelWrapper instance. If not None, given
            input is used over ModelWrapper object values.
        e.g. - nn.Softmax(dim=1), nn.Sigmoid()
    output_selection_func : 'round', 'argmax' or function reference, optional
        Note: Not required if model is a ModelWrapper instance.
        If round, prediction is rounded off to 0 or 1. Can be useful for
            binary classification.
        If argmax, prediction is converted from shape (batch_size, num_classes)
            to (batch_size) by selecting class with highest value from each
            batch.
        If function reference, the function should accept one parameter (the
            model output after passing/not passing through pred_func) and
            should return the modified value.
        Not required if model is a ModelWrapper instance. If not None, given
            input is used over ModelWrapper object values.

    Returns
    -------
    Tensor
        A tensor of prediction probabilities.
    Tensor
        A tensor of prediction class.
    Tensor
        A tensor of all labels.
        If labels are not available return tensor of null.
    """

    # Get details from ModelWrapper object
    if type(model) == ModelWrapper:
        if not pred_func:
            pred_func = model.state_data()['pred_func']

        if not output_selection_func:
            output_selection_func = model.state_data()['output_selection_func']

        model = model.model()

    model.eval()
    if torch.is_tensor(data):
        if len(data.shape) == 3:
            data = torch.unsqueeze(data, 0)

        # Move data to model device
        model_device = next(model.parameters()).device.type
        if model_device != data.device.type:
            data = to_device(data, model_device)

        out = model(data).cpu()
        if pred_func:
            out = pred_func(out)

        if output_selection_func == 'round':
            pred = torch.round(out)
        elif output_selection_func == 'argmax':
            pred = torch.argmax(out, dim=1)
        elif callable(output_selection_func):
            pred = output_selection_func(out)
        else:
            pred = out

        return out.cpu(), pred.int().cpu(), torch.tensor(float('nan')).cpu()
    else:
        out_list = pred_list = label_list = None
        for batch in data:
            # Check if batch contains labels
            if len(batch) == 2:
                xb, yb = batch
            else:
                xb = batch[0]
                yb = torch.tensor([float('nan')]*xb.size()[0]).cpu()

            # Add batch_size dimension if not present
            if len(xb.shape) == 3:
                xb = torch.unsqueeze(xb, 0)

            out = model(xb).cpu()
            if pred_func:
                out = pred_func(out)

            if output_selection_func == 'round':
                pred = torch.round(out)
            elif output_selection_func == 'argmax':
                pred = torch.argmax(out, dim=1)
            elif callable(output_selection_func):
                pred = output_selection_func(out)
            else:
                pred = out

            if out_list is None:
                out_list = out
                pred_list = pred
                label_list = yb
            else:
                out_list = torch.cat((out_list, out))
                pred_list = torch.cat((pred_list, pred))
                label_list = torch.cat((label_list, yb))
        return out_list.cpu(), pred_list.int().cpu(), label_list.int().cpu()
