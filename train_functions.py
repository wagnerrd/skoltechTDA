import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from collections import defaultdict
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing.label import LabelEncoder

from torch import Tensor, LongTensor
from torch.utils.data import DataLoader, Sampler
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR

from IPython.display import clear_output
import matplotlib.pyplot as plt


# def load minibatches
# можно просто итерироваться по созданным датасэмплерам для трейна и теста


def compute_labels(outputs):
    probs = F.softmax(outputs, dim=-1).cpu().data.numpy()
    return np.argmax(probs, axis=1)


def data_typing(batch_input, batch_targets, params):

    def tensor_cast(x):
        return x.cuda(params["cuda_device_id"]) if params["cuda"] else x.cpu()

    def cast(x):            
        if isinstance(x, torch.Tensor):
            return tensor_cast(x)
        elif isinstance(x, list):
            return [cast(v) for v in x]
        elif isinstance(x, dict):
            return {k: cast(v) for k, v in x.items()}
        elif isinstance(x, tuple):
            return tuple(cast(v) for v in x)
        else:
            return x

    batch_input = cast(batch_input)
    batch_targets = cast(batch_targets)

    return batch_input, batch_targets


def train_one_epoch(model, optimizer, train_data, params, criterion, variable_created_by_model):
    
    # training
    train_loss = []
    train_preds = []
    train_targets = []
    model.train(True)
    for i, (batch_input, batch_target) in enumerate(train_data, start=1):
        # transform input to tensor
        batch_input, batch_target = data_typing(batch_input, batch_target, params)

        if not variable_created_by_model:
            batch_input = Variable(batch_input)
        batch_target = Variable(batch_target)
        
        optimizer.zero_grad()
        batch_output = model(batch_input)
        loss = criterion(batch_output, batch_target)
        loss.backward()
        optimizer.step()
        
        train_loss.append(loss.item())
        train_preds.extend(list(compute_labels(batch_output)))
        train_targets.extend(list(batch_target.cpu().data.numpy()))
    
    return train_loss, train_preds, train_targets


def validate(model, val_data, params, criterion, variable_created_by_model):
    
    # validation
    val_loss = []
    val_preds = []
    val_targets = []
    model.train(False)
    for i, (batch_input, batch_target) in enumerate(val_data, start=1):
        # transform input to tensor
        batch_input, batch_target = data_typing(batch_input, batch_target, params)

        if not variable_created_by_model:
            batch_input = Variable(batch_input)
        batch_target = Variable(batch_target)
        
        batch_output = model(batch_input)
        loss = criterion(batch_output, batch_target)
        
        val_loss.append(loss.item())
        val_preds.extend(list(compute_labels(batch_output)))
        val_targets.extend(list(batch_target.cpu().data.numpy()))
    
    return val_loss, val_preds, val_targets


def train(model, optimizer, train_data, val_data, params, metric=accuracy_score, criterion=nn.CrossEntropyLoss(), variable_created_by_model=True):
    
    mean_train_loss = []
    mean_val_loss = []
    mean_train_metric = []
    mean_val_metric = []

    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.5 ** (epoch // params["lr_ep_step"]))
    
    for epoch in range(params["epochs"]):
        start_time = time.time()
        
        scheduler.step()
        print("current lr = {}".format(scheduler.get_lr()[0]))
        
        train_loss, train_preds, train_targets = train_one_epoch(
            model, optimizer, train_data, params, criterion, variable_created_by_model)
        val_loss, val_preds, val_targets = validate(
            model, val_data, params, criterion, variable_created_by_model)

        # print the results for this epoch:
        mean_train_loss.append(np.mean(train_loss))
        mean_val_loss.append(np.mean(val_loss))
        mean_train_metric.append(metric(train_targets, train_preds))
        mean_val_metric.append(metric(val_targets, val_preds))
        
        clear_output(True)
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.plot(mean_train_loss)
        plt.plot(mean_val_loss)
        plt.subplot(122)
        plt.plot(mean_train_metric)
        plt.plot(mean_val_metric)
        plt.gca().set_ylim([0, 1])
        plt.show()
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, params["epochs"], time.time() - start_time))
        print("  training loss (in-iteration): \t{:.6f}".format(mean_train_loss[-1]))
        print("  validation loss: \t\t\t{:.6f}".format(mean_val_loss[-1]))
        print("  training metric: \t\t\t{:.2f}".format(mean_train_metric[-1]))
        print("  validation metric: \t\t\t{:.2f}".format(mean_val_metric[-1]))
        
#         if mean_train_loss[-1] < epsilon:
#             break

    return mean_train_loss, mean_val_loss, mean_train_metric, mean_val_metric

# ? def cross_val_trains