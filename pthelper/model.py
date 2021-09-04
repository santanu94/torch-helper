import torch
from pathlib import Path

def __get_req_key(name):
    return '.'.join([n for n in name.split('.')[:2] if n not in ['weight', 'bias']])

def __get_layer_names(model):
    layer_names = []
    for name, _ in model.named_parameters():
        key = __get_req_key(name)
        if key not in layer_names: layer_names.append(key)

    return layer_names

def freeze_to(model, n_layers):
    layers_to_freeze = __get_layer_names(model)[: n_layers]

    for name, param in model.named_parameters():
        if __get_req_key(name) in layers_to_freeze:
            param.requires_grad = False
        else:
            param.requires_grad = True

def unfreeze(model):
    freeze_to(model, 0)

def save_model(model, filename, path='./'):
    torch.save(model.state_dict(), Path(path,filename))
        
def load_model(model, filename, path='./'):
    """Load saved model"""
    model_state_dict = torch.load(Path(path, filename))
    model.load_state_dict(model_state_dict)
