# for data transformation 
import numpy as np 
# for visualizing the data 
import matplotlib.pyplot as plt 
# for opening the media file 
import scipy.io.wavfile as wavfile
import cv2


import os
import time
from glob import glob
import pathlib

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from data import InfantDataset
from networkArchitecture import FeedForwardNet
from sklearn.model_selection import train_test_split

from utils import seeding, create_dir, epoch_time
from sklearn import preprocessing

import datetime
import numpy as np
from sklearn.utils import class_weight





def train(model, loader, optimizer, loss_fn, device = None):
    epoch_loss = 0.0
    epoch_acc = 0

    model.train()
    for x, y in loader:
        x = x# .to(device)#dtype=torch.long)
        y = y.to(dtype=torch.long)#,device
        
        optimizer.zero_grad()
        y_pred = model(x)
#         y_pred = torch.argmax(y_pred, axis = 1)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += (y == torch.argmax(y_pred, axis = 1)).sum()

    epoch_loss = epoch_loss/len(loader)
    epoch_acc = epoch_acc/len(loader.dataset)

    return epoch_loss, epoch_acc

def evaluate(model, loader, loss_fn, device = None):
    epoch_loss = 0.0
    epoch_acc = 0
    
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x# .to(device)#dtype=torch.float32)
            y = y.to(dtype=torch.long)#, device, )

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()
            epoch_acc += (y == torch.argmax(y_pred, axis = 1)).sum()
        epoch_loss = epoch_loss/len(loader)
        epoch_acc = epoch_acc/len(loader.dataset)
        torch.cuda.empty_cache()

        
    return epoch_loss, epoch_acc

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import numpy as np


def calc_instance_acc(y_true, y_pred, classes_num):
  #print(y_true)
  #print(y_pred)
  #y_true = y_true.tolist()
  #y_pred = y_pred.tolist()
  
  
  """
  cm = confusion_matrix(y_true, y_pred)
  cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  instance_acc = cm.diagonal()
  return instance_acc
  """
  #print(list(set(y_true)))
  #print(list(set(y_true)))
  return {name.tolist(): accuracy_score(np.array(y_true) == name.tolist(), np.array(y_pred) == name.tolist()) for i, name in enumerate(list(set(y_true)))}
  
  
  """
  
  """
  
  """
  instance_acc = []
  for c in range(classes_num):
    tmp = ((y_pred == y) * (y == c)).float() / max((y == c).sum(), 1)
    instance_acc.append(tmp)
  return instance_acc
  """


def inference(model, loader, loss_fn, device = None, classes_num = 5):
    epoch_loss = 0.0
    epoch_acc = 0
    classes_pred = []
    classes_gt = []
    
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x# .to(device)#dtype=torch.float32)
            y = y.to(dtype=torch.long)#, device, )

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()
            epoch_acc += (y == torch.argmax(y_pred, axis = 1)).sum()
            classes_gt.extend(y)
            classes_pred.extend(torch.argmax(y_pred, axis = 1))
        epoch_loss = epoch_loss/len(loader)
        epoch_acc = epoch_acc/len(loader.dataset)
        torch.cuda.empty_cache()
    instance_acc = calc_instance_acc(classes_gt, classes_pred, classes_num)
    #instance_acc = None
    return epoch_loss, epoch_acc, classes_gt, classes_pred, instance_acc


def fit(model, train_loader, valid_loader, optimizer, loss_fn, EPOCHS, checkpoint_path, device, valid_metric = "acc"):
 
  epoch_best = 0
  patient = 0
  
  if valid_metric == "acc":
    best_valid = -float("inf")
  elif valid_metric == "loss":
    best_valid = float("inf")
        
  
  
  history = {"loss":[], "acc":[], "val_loss":[], "val_acc":[], "epochs":[], "lr":[]}
  
  print("\n\n------------Start training-----------\n\n")
  for epoch in range(EPOCHS):
      start_time = time.time()
      text = ""
      loss, acc = train(model, train_loader, optimizer, loss_fn, device)
      val_loss, val_acc = evaluate(model, valid_loader, loss_fn, device)
  
      """ Saving the model """
      
      if valid_metric == "acc":
        if val_acc > best_valid:
          epoch_best = epoch
          best_valid = val_acc
          torch.save(model.state_dict(), checkpoint_path)
          text = f"----------------{best_valid*100:.3f} %"
          patient = 0
        else:
          patient += 1     
      elif valid_metric == "loss":
        pass

      data_str = f'Epoch:{epoch+1:02}|| Acc: {acc*100:.3f}  {val_acc*100:.3f} ----- Loss: {loss:.3f} , {val_loss:.3f}{text}\n'
      print(data_str)
      history["loss"].append(loss)
      history["acc"].append(acc)
      history["val_loss"].append(val_loss)
      history["val_acc"].append(val_acc)
      history["epochs"].append(epoch)
      history["lr"].append(optimizer.param_groups[-1]['lr'])
      if best_valid == 1.0 or patient == 50:
        break
        
  results = {"model": model, "history": history, "best_valid": best_valid, "epoch_best" : epoch_best}
  print("\n\n------------End training-----------\n\n")
  
  return results



def get_weights(y):
#     y = np.argmax(y_cat, axis = 1)

    # y_train = [1, 1,2,2,2,2,2,0,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1]
    classes = np.unique(y)
    weights = class_weight.compute_class_weight('balanced', classes=classes, y=y)
    class_weights = {k: v for k, v in zip(classes, weights)}
#     print('Class weights:', class_weights)
    return class_weights
