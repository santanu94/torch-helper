import torch
from .utils.metrics import accuracy
from .utils import get_default_device, to_device
import types

class ModelWrapper():
    def __init__(self, model, opt=None, criterion=None, device=None, **watch):
        self.__state_data = {}
        self.__state_data['model'] = model
        self.__state_data['opt'] = opt
        self.__state_data['criterion'] = criterion
        self.__state_data['total_trained_epochs'] = 0
        self.__state_data['best_val_loss'] = None
        self.__state_data['history'] = { 'train_loss': [], 'val_loss': [], 'val_acc': [] }
        self.__state_data['model'] = to_device(model, device if device else get_default_device())
        self.watch = watch

    # Setter methods
    def set_optimizer(self, opt):
        self.__state_data['opt'] = opt

    def set_criterion(self, criterion):
        self.__state_data['criterion'] = criterion

    # Getter methods
    def model(self):
        return self.__state_data['model']

    def optimizer(self):
        """Getter function for optimizer"""
        print(self.__state_data['opt'])

    def criterion(self):
        """Getter function for criterion"""
        print(self.__state_data['criterion'])

    def parameters(self):
        return self.__state_data['model'].parameters()

    def fit(self, epoch, train_dl, val_dl, test_dl=None, save_best_model_policy='val_loss'):
        """
        Train model on training data.

        Parameters
        ----------
        epoch : int
            The number of epochs to train the model, i.e., the number of times
            the model will train on the training data.
        train_dl : DataLoader, DataLoaderWrapper or Iterable
            Data on which the model will be trained.
        val_dl : DataLoader, DataLoaderWrapper or Iterable
            Data to be used for validation.
        test_dl : DataLoader, DataLoaderWrapper or Iterable, optional
            Data to be used to test model performance.
        save_best_model_policy : str or function or None, default='val_loss'
            Can be either 'val_loss', 'val_acc' or a function for custom save policy.
            If 'val_loss', model will be saved on every validation loss improvemrnt.
            if 'val_acc', model will be saved on every validation accuracy improvement.
            If a function, watch['var_name'] can be used within the body.
            If None, no model will be saved.
        """

        # Perform opt and criterion checks
        assert self.__state_data['opt'], 'Optimizer not defined! Please use set_optimizer() to set an optimizer.'
        assert self.__state_data['criterion'], 'Criterion not defined! Please use set_criterion() to set a criterion.'

        for i in range(epoch):
            train_loss_epoch_history = []
            val_loss_epoch_history = []
            val_acc_epoch_history = []
            for xb, yb in train_dl:
                xb = xb.float()
                yb = yb.float()

                self.__state_data['model'].train()
                out = self.__state_data['model'](xb).view(yb.shape)
                loss = self.__state_data['criterion'](out, yb)
                loss.backward()
                self.__state_data['opt'].step()
                self.__state_data['opt'].zero_grad()

                train_loss_epoch_history.append(loss.item())
                self.__state_data['total_trained_epochs'] += 1

            mean_epoch_train_loss = mean(train_loss_epoch_history)
            mean_epoch_val_loss, mean_epoch_val_acc = self.__validation_step(val_dl)
            self.__end_of_epoch_step(train_loss_epoch_history, mean_epoch_val_loss, mean_epoch_val_acc, save_best_model_policy)

            print('epoch ->', i+1, '  train loss ->', mean_epoch_train_loss, '  val loss ->', mean_epoch_val_loss, '  val acc ->', mean_epoch_val_acc)

    @torch.no_grad()
    def __validation_step(self, dl):
        """Perform validation step during training and return loss and accuracy for given epoch"""

        # Perform criterion checks
        assert self.__state_data['criterion'], 'Criterion not defined! Please use set_criterion() to set a criterion.'

        # ToDo: custom function injection
        val_loss_epoch_history = []
        val_acc_epoch_history = []
        for xb, yb in dl:
            xb = xb.float()
            yb = yb.float()

            self.__state_data['model'].eval()
            out = self.__state_data['model'](xb).view(yb.shape)
            loss = self.__state_data['criterion'](out, yb)

            val_loss_epoch_history.append(loss.item())
            val_acc_epoch_history.append(self.__get_output_label_similarity(nn.Sigmoid()(out), yb))
        return self.__mean(val_loss_epoch_history), accuracy(val_acc_epoch_history)

    def __end_of_epoch_step(self, mean_epoch_train_loss, mean_epoch_val_loss, mean_epoch_val_acc, save_best_model_policy):
        if save_best_model_policy and (not self.__state_data['best_val_loss'] or mean_epoch_val_loss < self.__state_data['best_val_loss']):
            torch.save(self.__state_data['model'].state_dict(), TMP_PATH+'checkpoint.pth')
            self.__state_data['best_val_loss'] = mean_epoch_val_loss

        self.__state_data['history']['train_loss'].append(mean_epoch_train_loss)
        self.__state_data['history']['val_loss'].append(mean_epoch_val_loss)
        self.__state_data['history']['val_acc'].append(mean_epoch_val_acc)

    def __save_best_model(self, mean_epoch_val_loss, mean_epoch_val_acc, save_best_model_policy):
        if save_best_model_policy == 'val_loss':
            if not self.__state_data['best_val_loss'] or mean_epoch_val_loss < self.__state_data['best_val_loss']:
                torch.save(self.__state_data['model'].state_dict(), TMP_PATH+'checkpoint.pth')
                self.__state_data['best_val_loss'] = mean_epoch_val_loss
        elif save_best_model_policy == 'val_acc':
            pass
        elif isinstance(save_best_model_policy, types.FunctionType):
            save_best_model_policy()

    def __get_output_label_similarity(self, output, labels):
        return (torch.round(output) == labels)

    def __mean(self, list_var):
        """Return the average of all values in a list"""
        return sum(list_var) / len(list_var)

    def untrained_model_stats(self, val_dl):
        mean_epoch_val_loss, mean_epoch_val_acc = self.__validation_step(val_dl)
        print('initial val loss ->', mean_epoch_val_loss, '  initial val acc ->', mean_epoch_val_acc)

    def model_summary(self):
        print(self.__state_data['model'])
