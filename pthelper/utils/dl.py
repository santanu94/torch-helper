from .device import get_default_device, to_device

class DataLoaderWrapper():
    def __init__(self, dl, device=None):
        self.dl = dl
        if device:
            self.device = device
        else:
            self.device = get_default_device()

    def __iter__(self):
        for batch in self.dl:
            yield to_device(batch, self.device)

    def __len__(self):
        return len(self.dl)
