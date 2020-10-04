import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.nn.functional import one_hot
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt
from .utils.metrics import get_accuracy, get_f1_score
from .utils import get_default_device, to_device
from pathlib import Path
import os
import types
import warnings

class ModelWrapper():
    def __init__(self, model, pred_func=None, output_selection_func=None, device=None, **watch):
        """
        Wrapper class for model, optimizer and loss.

        Parameters
        ----------
        model : Pytorch Model
        pred_func : Pytorch layer or function reference, optional
            If model outputs logit then pass a function reference or a pytorch
            layer to get prediction or calculate model accuracy.
            For regression output, keep None.
            e.g. - nn.Softmax(dim=1), nn.Sigmoid()
        output_selection_func : 'round', 'argmax' or function reference, optional
            If round, prediction is rounded off to 0 or 1. Can be useful for
                binary classification.
            If argmax, prediction is converted from shape (batch_size, num_classes)
                to (batch_size) by selecting class with highest value from each
                batch.
            If function reference, the function should accept one parameter (the
                model output after passing/not passing through pred_func) and
                should return the modified value.
        device : 'cpu' or 'cuda', optional
            If not None, the model is moved to the given device.
            If None, use GPU if available or else use CPU.
        watch : array or dict
            Additional parameters to track. Can be accessed from anywhere using
                <ModelWrapper_object>.watch.
        """
        self.__state_data = {}
        self.__state_data['model'] = model
        self.__state_data['pred_func'] = pred_func
        # self.__state_data['one_hot_target'] = one_hot_target
        # self.__state_data['num_classes'] = num_classes
        self.__state_data['output_selection_func'] = output_selection_func
        self.__state_data['total_trained_epochs'] = 0
        self.__state_data['best_val_loss'] = None
        self.__state_data['batch_model_pred'] = None
        self.__state_data['batch_true_labels'] = None
        self.__state_data['history'] = { 'epoch': None, 'lr': None, 'train_loss': None, 'train_acc': None, 'train_f1': None, 'val_loss': None, 'val_acc': None, 'val_f1': None }
        self.__state_data['model'] = to_device(model, device if device else get_default_device())
        self.watch = watch

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

    def state_data(self):
        return self.__state_data

    # Train model
    def fit(self, epoch, train_dl, val_dl, optimizer, criterion, scheduler=None, grad_clip=None, f1_score=None, save_best_model_policy='val_loss', save_best_model_path='model'):
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
        optimizer : Pytorch optimizer
            A valid pytorch optimizer from torch.optim, e.g., torch.optim.SGD(...).
        criterion : Pytorch criterion
            A valid pytorch criterion, a.k.a., the loss function. e.g. nn.CrossEntropyLoss().
        scheduler : Pytorch scheduler, optional
            Scheduler to change LR dynamically.
            Note - CosineAnnealingLR and CosineAnnealingWarmRestarts are not supported right now.
        grad_clip : float or int, optional
            If provided, clip gradients of model before performin optimizer.step().
        f1_score : 'binary', 'micro', 'macro', 'weighted', 'samples' or None, optional
            Calculate and display f1 score during training. Implements sklearn.metrics.f1_score
            'binary'
                Only report results for the class specified by pos_label. This is applicable only
                if targets (y_{true,pred}) are binary.
            'micro'
                Calculate metrics globally by counting the total true positives, false negatives
                and false positives.
            'macro'
                Calculate metrics for each label, and find their unweighted mean. This does not
                take label imbalance into account.
            'weighted'
                Calculate metrics for each label, and find their average weighted by support
                (the number of true instances for each label). This alters ‘macro’ to account for
                label imbalance; it can result in an F-score that is not between precision and recall.
            'samples'
                Calculate metrics for each instance, and find their average (only meaningful for
                multilabel classification where this differs from accuracy_score).
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
        assert optimizer, 'Optimizer not defined!'
        assert criterion, 'Criterion not defined!'

        # Add to state data
        self.__state_data['opt'] = optimizer
        self.__state_data['criterion'] = criterion
        self.__state_data['scheduler'] = scheduler
        self.__state_data['grad_clip'] = grad_clip
        self.__state_data['f1_score'] = f1_score
        self.__state_data['save_best_model_policy'] = save_best_model_policy
        self.__state_data['save_best_model_path'] = Path(save_best_model_path)

        # Create best model store directory (recursive)
        if self.__state_data['save_best_model_policy']: os.makedirs(save_best_model_path, exist_ok=True)

        # Initialize history
        self.__init_history(epoch)

        for i in range(1, epoch+1):
            train_loss_epoch_history = None
            train_epoch_pred = None
            train_epoch_true_label = None
            for xb, yb in train_dl:
                xb = xb.float()
                yb = yb.float()

                self.__state_data['model'].train()
                out = self.__state_data['model'](xb)

                if type(self.__state_data['criterion']) == nn.BCEWithLogitsLoss:
                    out = out.view(yb.shape)
                elif type(self.__state_data['criterion']) == nn.CrossEntropyLoss:
                    yb = yb.long()

                loss = self.__state_data['criterion'](out, yb)
                loss.backward()

                if self.__state_data['grad_clip']:
                    clip_grad_norm_(self.__state_data['model'].parameters(), self.__state_data['grad_clip'])

                self.__state_data['opt'].step()
                self.__state_data['opt'].zero_grad()

                if self.__state_data['pred_func']:
                    out = self.__state_data['pred_func'](out)

                if self.__state_data['output_selection_func'] == 'round':
                    out = torch.round(out)
                elif self.__state_data['output_selection_func'] == 'argmax':
                    out = torch.argmax(out, dim=1)
                elif callable(self.__state_data['output_selection_func']):
                    out = output_selection_func(out)

                if train_loss_epoch_history is None:
                    train_loss_epoch_history = loss.detach().view(1)
                    train_epoch_pred = out
                    train_epoch_true_label = yb
                else:
                    train_loss_epoch_history = torch.cat((train_loss_epoch_history, loss.detach().view(1)))
                    train_epoch_pred = torch.cat((train_epoch_pred, out))
                    train_epoch_true_label = torch.cat((train_epoch_true_label, yb))

                if self.__state_data['scheduler'] and isinstance(scheduler, (lr_scheduler.CyclicLR, lr_scheduler.OneCycleLR)):
                    self.__state_data['scheduler'].step()

            train_epoch_true_label = train_epoch_true_label.view(-1)
            train_epoch_pred = train_epoch_pred.view(-1)

            mean_epoch_train_loss = torch.mean(train_loss_epoch_history).item()
            mean_epoch_train_acc = get_accuracy(train_epoch_true_label, train_epoch_pred)
            train_f1_score = get_f1_score(train_epoch_true_label, train_epoch_pred, average=self.__state_data['f1_score']) if self.__state_data['f1_score'] else None
            mean_epoch_val_loss, mean_epoch_val_acc, val_f1_score = self.__validation_step(val_dl, self.__state_data['f1_score'])

            self.__end_of_epoch_step(i, mean_epoch_train_loss, mean_epoch_train_acc, train_f1_score, mean_epoch_val_loss, mean_epoch_val_acc, val_f1_score)

    def __init_history(self, num_epochs):
        if self.__state_data['total_trained_epochs'] == 0:
            self.__state_data['history']['epoch'] = torch.empty(num_epochs, dtype=torch.int16)
            self.__state_data['history']['lr'] = torch.empty(num_epochs, dtype=torch.float32)
            self.__state_data['history']['train_loss'] = torch.empty(num_epochs, dtype=torch.float32)
            self.__state_data['history']['train_acc'] = torch.empty(num_epochs, dtype=torch.float32)
            self.__state_data['history']['val_loss'] = torch.empty(num_epochs, dtype=torch.float32)
            self.__state_data['history']['val_acc'] = torch.empty(num_epochs, dtype=torch.float32)
            self.__state_data['history']['train_f1'] = torch.empty(num_epochs, dtype=torch.float32)
            self.__state_data['history']['val_f1'] = torch.empty(num_epochs, dtype=torch.float32)
        if self.__state_data['total_trained_epochs'] > 0:
            self.__state_data['history']['epoch'] = torch.cat((self.__state_data['history']['epoch'], torch.empty(num_epochs, dtype=torch.int16)))
            self.__state_data['history']['lr'] = torch.cat((self.__state_data['history']['lr'], torch.empty(num_epochs, dtype=torch.float32)))
            self.__state_data['history']['train_loss'] = torch.cat((self.__state_data['history']['train_loss'], torch.empty(num_epochs, dtype=torch.float32)))
            self.__state_data['history']['train_acc'] = torch.cat((self.__state_data['history']['train_acc'], torch.empty(num_epochs, dtype=torch.float32)))
            self.__state_data['history']['val_loss'] = torch.cat((self.__state_data['history']['val_loss'], torch.empty(num_epochs, dtype=torch.float32)))
            self.__state_data['history']['val_acc'] = torch.cat((self.__state_data['history']['val_acc'], torch.empty(num_epochs, dtype=torch.float32)))
            self.__state_data['history']['train_f1'] = torch.cat((self.__state_data['history']['train_f1'], torch.empty(num_epochs, dtype=torch.float32)))
            self.__state_data['history']['val_f1'] = torch.cat((self.__state_data['history']['val_f1'], torch.empty(num_epochs, dtype=torch.float32)))

    @torch.no_grad()
    def __validation_step(self, dl, f1_score):
        """Perform validation step during training and return loss and accuracy for given epoch"""

        # Perform criterion checks
        assert self.__state_data['criterion'], 'Criterion not defined! Please use set_criterion() to set a criterion.'

        # ToDo: custom function injection
        val_loss_epoch_history = None
        # val_acc_epoch_history = None
        val_epoch_pred = None
        val_epoch_true_label = None
        for xb, yb in dl:
            xb = xb.float()
            yb = yb.float()

            self.__state_data['model'].eval()
            out = self.__state_data['model'](xb)

            if type(self.__state_data['criterion']) == nn.BCEWithLogitsLoss:
                out = out.view(yb.shape)
            elif type(self.__state_data['criterion']) == nn.CrossEntropyLoss:
                yb = yb.long()

            loss = self.__state_data['criterion'](out, yb)

            if self.__state_data['pred_func']:
                out = self.__state_data['pred_func'](out)

            if self.__state_data['output_selection_func'] == 'round':
                out = torch.round(out)
            elif self.__state_data['output_selection_func'] == 'argmax':
                out = torch.argmax(out, dim=1)
            elif callable(self.__state_data['output_selection_func']):
                out = output_selection_func(out)

            if val_loss_epoch_history is None:
                val_loss_epoch_history = loss.detach().view(1)
                val_epoch_pred = out
                val_epoch_true_label = yb
            else:
                val_loss_epoch_history = torch.cat((val_loss_epoch_history, loss.detach().view(1)))
                val_epoch_pred = torch.cat((val_epoch_pred, out))
                val_epoch_true_label = torch.cat((val_epoch_true_label, yb))

        val_epoch_true_label = val_epoch_true_label.view(-1)
        val_epoch_pred = val_epoch_pred.view(-1)
        return torch.mean(val_loss_epoch_history).item(), get_accuracy(val_epoch_true_label, val_epoch_pred), get_f1_score(val_epoch_true_label, val_epoch_pred, average=f1_score) if f1_score else None

    def __end_of_epoch_step(self, epoch, mean_epoch_train_loss, mean_epoch_train_acc, train_f1_score, mean_epoch_val_loss, mean_epoch_val_acc, val_f1_score):
        if self.__state_data['save_best_model_policy']:
            self.__save_best_model(mean_epoch_val_loss, mean_epoch_val_acc)

        # Get lr used in the epoch
        for param_group in self.__state_data['opt'].param_groups:
            lr = param_group['lr']

        # Record training history
        self.__state_data['history']['epoch'][self.__state_data['total_trained_epochs']] = epoch
        self.__state_data['history']['lr'][self.__state_data['total_trained_epochs']] = lr
        self.__state_data['history']['train_loss'][self.__state_data['total_trained_epochs']] = mean_epoch_train_loss
        self.__state_data['history']['train_acc'][self.__state_data['total_trained_epochs']] = mean_epoch_train_acc
        self.__state_data['history']['val_loss'][self.__state_data['total_trained_epochs']] = mean_epoch_val_loss
        self.__state_data['history']['val_acc'][self.__state_data['total_trained_epochs']] = mean_epoch_val_acc
        if val_f1_score is not None:
            self.__state_data['history']['train_f1'][self.__state_data['total_trained_epochs']] = train_f1_score
            self.__state_data['history']['val_f1'][self.__state_data['total_trained_epochs']] = val_f1_score
        else:
            self.__state_data['history']['train_f1'][self.__state_data['total_trained_epochs']] = float('NaN')
            self.__state_data['history']['val_f1'][self.__state_data['total_trained_epochs']] = float('NaN')

        # Step scheduler
        if self.__state_data['scheduler']:
            if isinstance(self.__state_data['scheduler'], (lr_scheduler.LambdaLR, lr_scheduler.MultiplicativeLR, lr_scheduler.StepLR, lr_scheduler.MultiStepLR, lr_scheduler.ExponentialLR)):
                self.__state_data['scheduler'].step()
            elif isinstance(self.__state_data['scheduler'], lr_scheduler.ReduceLROnPlateau):
                self.__state_data['scheduler'].step(mean_epoch_train_loss)

        self.__state_data['total_trained_epochs'] += 1

        #print report
        report = f'epoch -> {epoch}    lr -> {lr:.6f}    train loss -> {mean_epoch_train_loss:.6f}    train acc -> {mean_epoch_train_acc:.6f}'
        if train_f1_score is not None:
            report += f'    train_f1 -> {train_f1_score:.6f}'
        report += f'    val loss -> {mean_epoch_val_loss:.6f}    val acc -> {mean_epoch_val_acc:.6f}'
        if val_f1_score is not None:
            report += f'    val_f1 -> {val_f1_score:.6f}'
        print(report)

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

    def __get_req_key(self, name):
        return '.'.join([n for n in name.split('.')[:2] if n not in ['weight', 'bias']])

    def __get_layer_names(self):
        layer_names = []
        for name, _ in self.__state_data['model'].named_parameters():
            key = self.__get_req_key(name)
            if key not in layer_names: layer_names.append(key)

        return layer_names

    def freeze_to(self, n_layers):
        layers_to_freeze = self.__get_layer_names()[: n_layers]

        for name, param in self.__state_data['model'].named_parameters():
            if self.__get_req_key(name) in layers_to_freeze:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def unfreeze(self):
        self.freeze_to(0)

    def performance_stats(self, val_dl, f1_score=None):
        """Print loss and accuracy of model"""
        mean_epoch_val_loss, mean_epoch_val_acc, val_f1_score = self.__validation_step(val_dl, f1_score)
        report = f'loss -> {mean_epoch_val_loss:.6f}    acc -> {mean_epoch_val_acc:.6f}'
        if f1_score:
            report += f'    f1 -> {val_f1_score:.6f}'
        print(report)

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

    def plot_lr(self):
        """Plot graph of learning rate used in each epoch"""
        assert len(self.__state_data['history']['lr']) > 0, 'Model must be trained first.'

        plt.plot(range(1, self.__state_data['total_trained_epochs']+1), self.__state_data['history']['lr'].tolist(), label = 'Learning Rate')
        plt.xlabel('epochs')
        plt.ylabel('value')
        plt.legend()
        plt.show()

    def training_history(self):
        """Return dictionary containing training and validation loss and accuracy captured during training"""
        return self.__state_data['history']

    def load_bestmodel(self):
        """Load best saved model if save_best_model_policy is 'val_loss' or 'val_acc'"""
        if 'save_best_model_policy' not in self.__state_data:
            print('Model has not been trained yet.')
        elif not self.__state_data['save_best_model_policy']:
            print('No model to load!! Best model was not saved while training.')
        elif isinstance(self.__state_data['save_best_model_policy'], str):
            model_state_dict = torch.load(self.__state_data['save_best_model_path'] / 'bestmodel.pth')
            self.__state_data['model'].load_state_dict(model_state_dict)
        else:
            print('Unclear save best model policy.')

    def model_summary(self):
        """Print model"""
        print(self.__state_data['model'])
