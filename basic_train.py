class ModelWrapper():
    def __init__(self, model, device=None):
        self.__model = model
        if device:
            self.__model = to_device(model, device)
        else:
            self.__model = to_device(model, get_default_device())

    def set_optimizer(self, opt):
        self.__opt = opt

    def set_criterion(self, criterion):
        self.__criterion = criterion

    def fit(self, lr, epoch):
        for i in range(epoch):
            train_loss_epoch_history = []
            val_loss_epoch_history = []
            val_acc_epoch_history = []
            for xb, yb in train_data_loader:
                xb = xb.float()
                yb = yb.float()

                self.__model.train()
                out = self.__model(xb).squeeze()
                loss = self.__criterion(out, yb)
                loss.backward()
                self.__opt.step()
                self.__opt.zero_grad()

                train_loss_epoch_history.append(loss.item())

            mean_epoch_train_loss = mean(train_loss_epoch_history)
            mean_epoch_val_loss, mean_epoch_val_acc = self.__validation_step(val_data_loader)
            end_of_epoch_step(self.__model, train_loss_epoch_history, mean_epoch_val_loss, mean_epoch_val_acc)

            print('epoch ->', i+1, '  train loss ->', mean_epoch_train_loss, '  val loss ->', mean_epoch_val_loss, '  val acc ->', mean_epoch_val_acc)

    @torch.no_grad()
    def __validation_step(self, dl):
        val_loss_epoch_history = []
        val_acc_epoch_history = []
        for xb, yb in dl:
            xb = xb.float()
            yb = yb.float()

            self.__model.eval()
            out = self.__model(xb).view(yb.shape)
            loss = self.__criterion(out, yb)

            val_loss_epoch_history.append(loss.item())
            val_acc_epoch_history.append(get_output_label_similarity(nn.Sigmoid()(out), yb))
        return mean(val_loss_epoch_history), get_accuracy(val_acc_epoch_history)

    def untrained_model_stats(self):
        pass
