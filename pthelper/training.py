import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from .utils.metrics import accuracy
from .utils import get_default_device, to_device
from pathlib import Path
import os
import types

class ModelWrapper():
    def __init__(self, model, opt=None, criterion=None, device=None, **watch):
        self.__state_data = {}
        self.__state_data['model'] = model
        self.__state_data['opt'] = opt
        self.__state_data['criterion'] = criterion
        self.__state_data['total_trained_epochs'] = 0
        self.__state_data['best_val_loss'] = None
        self.__state_data['history'] = { 'epoch': None, 'train_loss': None, 'train_acc': None, 'val_loss': None, 'val_acc': None }
        self.__state_data['model'] = to_device(model, device if device else get_default_device())
        self.watch = watch

    # Setter methods
    def set_optimizer(self, opt):
        self.__state_data['opt'] = opt

    def set_criterion(self, criterion):
        self.__state_data['criterion'] = criterion

    # Getter methods
    def optimizer(self):
        """Getter function for optimizer"""
        print(self.__state_data['opt'])

    def criterion(self):
        """Getter function for criterion"""
        print(self.__state_data['criterion'])

    def model(self):
        """Return model object"""
        return self.__state_data['model']

    def parameters(self):
        return self.__state_data['model'].parameters()

    def fit(self, epoch, train_dl, val_dl, test_dl=None, save_best_model_policy='val_loss', save_best_model_path='model'):
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
        save_best_model_path : str, default='model'
            Location to save best model checkpoint.
            Not used if save_best_model_policy is None or a function.
            Manually save model if save_best_model_policy is a function.
        """

        # Perform opt and criterion checks
        assert self.__state_data['opt'], 'Optimizer not defined! Please use set_optimizer() to set an optimizer.'
        assert self.__state_data['criterion'], 'Criterion not defined! Please use set_criterion() to set a criterion.'

        # Add to state data
        self.__state_data['save_best_model_policy'] = save_best_model_policy
        self.__state_data['save_best_model_path'] = Path(save_best_model_path)

        # Create best model store directory (recursive)
        if self.__state_data['save_best_model_policy']: os.makedirs(save_best_model_path, exist_ok=True)

        for i in range(1, epoch+1):
            train_loss_epoch_history = None
            train_acc_epoch_history = []
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

                if train_loss_epoch_history is None:
                    train_loss_epoch_history = loss.detach().view(1)
                    train_acc_epoch_history = torch.round(nn.Sigmoid()(out)) == yb
                else:
                    train_loss_epoch_history = torch.cat((train_loss_epoch_history, loss.detach().view(1)))
                    train_acc_epoch_history = torch.cat((train_acc_epoch_history, torch.round(nn.Sigmoid()(out)) == yb))

            mean_epoch_train_loss = torch.mean(train_loss_epoch_history).cpu()
            mean_epoch_train_acc = accuracy(train_acc_epoch_history).cpu()
            mean_epoch_val_loss, mean_epoch_val_acc = self.__validation_step(val_dl)
            self.__end_of_epoch_step(i, mean_epoch_train_loss, mean_epoch_train_acc, mean_epoch_val_loss, mean_epoch_val_acc)

            print('epoch ->', i, '  train loss ->', mean_epoch_train_loss.item(), '  train acc ->', mean_epoch_train_acc.item(), '  val loss ->', mean_epoch_val_loss.item(), '  val acc ->', mean_epoch_val_acc.item())

    @torch.no_grad()
    def __validation_step(self, dl):
        """Perform validation step during training and return loss and accuracy for given epoch"""

        # Perform criterion checks
        assert self.__state_data['criterion'], 'Criterion not defined! Please use set_criterion() to set a criterion.'

        # ToDo: custom function injection
        val_loss_epoch_history = None
        val_acc_epoch_history = None
        for xb, yb in dl:
            xb = xb.float()
            yb = yb.float()

            self.__state_data['model'].eval()
            out = self.__state_data['model'](xb).view(yb.shape)
            loss = self.__state_data['criterion'](out, yb)

            if val_loss_epoch_history is None:
                val_loss_epoch_history = loss.detach().view(1)
                val_acc_epoch_history = torch.round(nn.Sigmoid()(out)) == yb
            else:
                val_loss_epoch_history = torch.cat((val_loss_epoch_history, loss.detach().view(1)))
                val_acc_epoch_history = torch.cat((val_acc_epoch_history, torch.round(nn.Sigmoid()(out)) == yb))

        return torch.mean(val_loss_epoch_history).cpu(), accuracy(val_acc_epoch_history).cpu()

    def __end_of_epoch_step(self, epoch, mean_epoch_train_loss, mean_epoch_train_acc, mean_epoch_val_loss, mean_epoch_val_acc):
        if self.__state_data['save_best_model_policy']:
            self.__save_best_model(mean_epoch_val_loss.item(), mean_epoch_val_acc.item())

        if self.__state_data['history']['epoch'] is None:
            self.__state_data['history']['epoch'] = torch.tensor(epoch).view(1)
            self.__state_data['history']['train_loss'] = mean_epoch_train_loss.view(1)
            self.__state_data['history']['train_acc'] = mean_epoch_train_acc.view(1)
            self.__state_data['history']['val_loss'] = mean_epoch_val_loss.view(1)
            self.__state_data['history']['val_acc'] = mean_epoch_val_acc.view(1)
        else:
            self.__state_data['history']['epoch'] = torch.cat((self.__state_data['history']['epoch'], torch.tensor(epoch).view(1)))
            self.__state_data['history']['train_loss'] = torch.cat((self.__state_data['history']['train_loss'], mean_epoch_train_loss.view(1)))
            self.__state_data['history']['train_acc'] = torch.cat((self.__state_data['history']['train_acc'], mean_epoch_train_acc.view(1)))
            self.__state_data['history']['val_loss'] = torch.cat((self.__state_data['history']['val_loss'], mean_epoch_val_loss.view(1)))
            self.__state_data['history']['val_acc'] = torch.cat((self.__state_data['history']['val_acc'], mean_epoch_val_acc.view(1)))
        self.__state_data['total_trained_epochs'] += 1

    def __save_best_model(self, mean_epoch_val_loss, mean_epoch_val_acc):
        if self.__state_data['save_best_model_policy'] == 'val_loss':
            if not self.__state_data['best_val_loss'] or mean_epoch_val_loss < self.__state_data['best_val_loss']:
                torch.save(self.__state_data['model'].state_dict(), self.__state_data['save_best_model_path'] / 'bestmodel.pth')
                self.__state_data['best_val_loss'] = mean_epoch_val_loss
        elif self.__state_data['save_best_model_policy'] == 'val_acc':
            pass
        elif isinstance(self.__state_data['save_best_model_policy'], types.FunctionType):
            save_best_model_policy()

    def __get_output_label_similarity(self, output, labels):
        return (torch.round(output) == labels)

    def __mean(self, list_var):
        """Return the average of all values in a list"""
        return sum(list_var) / len(list_var)

    def performance_stats(self, val_dl):
        """Print loss and accuracy of model"""
        mean_epoch_val_loss, mean_epoch_val_acc = self.__validation_step(val_dl)
        print('loss ->', mean_epoch_val_loss.item(), '  acc ->', mean_epoch_val_acc.item())

    def plot_loss(self):
        """Plot graph comparing training and validation loss"""
        assert len(self.__state_data['history']['train_loss']) > 0, 'Model must be trained first.'

        plt.plot(range(1, self.__state_data['total_trained_epochs']+1), self.__state_data['history']['train_loss'].tolist(), label = 'Training Loss')
        plt.plot(range(1, self.__state_data['total_trained_epochs']+1), self.__state_data['history']['val_loss'].tolist(), label = 'Validation Loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.show()

    def plot_acc(self):
        """Plot graph comparing training and validation accuracy"""
        assert len(self.__state_data['history']['train_acc']) > 0, 'Model must be trained first.'

        plt.plot(range(1, self.__state_data['total_trained_epochs']+1), self.__state_data['history']['train_acc'].tolist(), label = 'Training Aaccuracy')
        plt.plot(range(1, self.__state_data['total_trained_epochs']+1), self.__state_data['history']['val_acc'].tolist(), label = 'Validation Accuracy')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.legend()
        plt.show()

    def training_history(self):
        """Return dictionary containing training and validation loss and accuracy captured during training"""
        return self.__state_data['history']

    def load_bestmodel(self):
        """Load best saved model if save_best_model_policy is 'val_loss' or 'val_acc'"""
        if isinstance(self.__state_data['save_best_model_policy'], str):
            model_state_dict = torch.load(self.__state_data['save_best_model_path'] / 'bestmodel.pth')
            self.__state_data['model'].load_state_dict(model_state_dict)

    def model_summary(self):
        """Print model"""
        print(self.__state_data['model'])
