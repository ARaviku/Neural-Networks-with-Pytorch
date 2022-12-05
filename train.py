
import torch
import numpy as np
import random
import checkpoint
from dataset import DogDataset, DogCatDataset
from model import CNN
from plot import Plotter
import torch.nn as nn

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def predictions(logits):
    pred , prediction_value = torch.max(logits, dim=1)
    return prediction_value


def accuracy(y_true, y_pred):
    a= 0;
    b = 0;
    for i in range(y_true.numel()):
        if y_pred[i] == y_true[i]:
          a += 1
        b += 1
    accuracy_value = (a/b)*100
    return accuracy_value



def _train_epoch(train_loader, model, criterion, optimizer):
    """
    Train the model for one iteration through the train set.
    """
    for i, (X, y) in enumerate(train_loader):
        # clear parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()


def _evaluate_epoch(plotter, train_loader, val_loader, model, criterion, epoch):
    """
    Evaluates the model on the train and validation set.
    """
    stat = []
    for data_loader in [val_loader, train_loader]:
        y_true, y_pred, running_loss = evaluate_loop(data_loader, model, criterion)
        total_loss = np.sum(running_loss) / y_true.size(0)
        total_acc = accuracy(y_true, y_pred)
        stat += [total_acc, total_loss]
    plotter.stats.append(stat)
    plotter.log_cnn_training(epoch)
    plotter.update_cnn_training_plot(epoch)


def evaluate_loop(data_loader, model, criterion=None):
    model.eval()
    y_true, y_pred, running_loss = [], [], []
    for X, y in data_loader:
        with torch.no_grad():
            output = model(X)
            predicted = predictions(output.data)
            y_true.append(y)
            y_pred.append(predicted)
            if criterion is not None:
                running_loss.append(criterion(output, y).item() * X.size(0))
    model.train()
    y_true, y_pred = torch.cat(y_true), torch.cat(y_pred)
    return y_true, y_pred, running_loss


def train(config, dataset, model):
    # Data loaders
    train_loader, val_loader = dataset.train_loader, dataset.val_loader

    if 'use_weighted' not in config:
        criterion = nn.CrossEntropyLoss()
    else:
        # TODO (part h): define weighted loss function
        x = torch.empty(2)
        x[0] = 20
        x[1] =1
        criterion = nn.CrossEntropyLoss(weight = x)

    learning_rate = config['learning_rate']
    momentum = config['momentum']
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum =momentum)

    print('Loading model...')
    force = config['ckpt_force'] if 'ckpt_force' in config else False
    model, start_epoch, stats = checkpoint.restore_checkpoint(model, config['ckpt_path'], force=force)

    plot_name = config['plot_name'] if 'plot_name' in config else 'CNN'
    plotter = Plotter(stats, plot_name)

    _evaluate_epoch(plotter, train_loader, val_loader, model, criterion, start_epoch)

    for epoch in range(start_epoch, config['num_epoch']):
        _train_epoch(train_loader, model, criterion, optimizer)
        _evaluate_epoch(plotter, train_loader, val_loader, model, criterion, epoch + 1)
        checkpoint.save_checkpoint(model, epoch + 1, config['ckpt_path'], plotter.stats)

    print('Finished Training')

    # Save figure and keep plot open
    plotter.save_cnn_training_plot()
    plotter.hold_training_plot()


if __name__ == '__main__':
    # define config parameters for training
    config = {
        'dataset_path': 'data/images/dogs',
        'batch_size': 4,
        'if_resize': True,             # If resize of the image is needed
        'ckpt_path': 'checkpoints/cnn',  # directory to save our model checkpoints
        'num_epoch': 10,                 # number of epochs for training
        'learning_rate': 1e-3,           # learning rate
        'momentum': 0.9,                  # momentum
    }

    dataset = DogDataset(config['batch_size'], config['dataset_path'],config['if_resize'])
    model = CNN()
    train(config, dataset, model)
