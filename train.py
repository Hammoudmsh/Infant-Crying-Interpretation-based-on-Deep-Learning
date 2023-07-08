#!/usr/bin/env python -u









print("hhhhhhhhhhhhh")

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

import pandas as pd

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
import random

from utils import plotGraphs, append2csv
from ML_DL_utilis import MLDL_utilitis
from model import fit, inference, get_weights

from sklearn.metrics import classification_report

mldl_uts = MLDL_utilitis()
import parsing_file2



def readParametersFromCmd():
    global DATASET, EPOCHS, LEARNING_RATE, EARLY_STOPPING, FILES_TO_READ, BATCH_SIZE, OUTPUT_FILE, EXPERIMENT_TAG
    parser = parsing_file2.create_parser_disease_model()
    args = parser.parse_args()
    
    
    EPOCHS = args.epochs
    EARLY_STOPPING = args.es
    LEARNING_RATE  = args.lr
    BATCH_SIZE = args.batch_size
    FILES_TO_READ = args.file2read   
    if FILES_TO_READ == 0:
      FILES_TO_READ = -1 
    OUTPUT_FILE = "" + args.output.strip("''")
    print(OUTPUT_FILE)
    DATASET = "" + args.dataset.strip("''")
    EXPERIMENT_TAG = args.experiment_tag
    print(DATASET)
    

# torch.cuda.empty_cache()
# torch.cuda.is_available()
# # torch.cuda.device_count(),torch.cuda.current_device(), torch.cuda.device(0), torch.cuda.get_device_name(0)
#class_weights = get_weights(np.array([1, 1, 2, 2, 2, 3]).astype('float32'))
#class_weights




readParametersFromCmd()
current_model = datetime.datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
current_model = f"Results/model_{current_model}_{OUTPUT_FILE}/"
pathlib.Path(f'{current_model}/').mkdir(parents=True, exist_ok=True)#metrics
mldl_uts.setDir(d = f"{current_model}/")
checkpoint_path = f"{current_model}checkpoint.pth"



""" Repreducde results(seeding) """
seeding(42)


device = torch.device('cpu')
#DATASET = "aug_dataset/"


""" Load dataset """
data = sorted(list(pathlib.Path(DATASET).rglob("*.png")))[0:FILES_TO_READ]
random.shuffle(data)

classes =[str(fn.parent).split("/")[2] for fn in data]
classes_names = list(set(classes))


x = cv2.imread(str(data[0]))
H, W = x.shape[0:2]
shape = (1, 1, H, W)



"""Envode target"""
le = preprocessing.LabelEncoder()
le.fit(list(set(classes)))
classes = le.transform(classes)

"""Solve the issue of immbalanced data"""
class_weights = get_weights(np.array(classes).astype('float32'))
classesNum = len(set(classes))



"""Split data"""
train_x, valid_test_x, train_y, valid_test_y = train_test_split(data, classes, test_size=0.2, stratify = classes)
valid_x, test_x, valid_y, test_y = train_test_split(valid_test_x, valid_test_y, test_size=0.1, stratify = valid_test_y)



print("----------------------------------------------------Information---------------------------------------------")
print("The model saved in ", current_model)

print("Split size: ", len(train_x), len(valid_x), len(test_x))
print("classes_names_codes", classes_names)
print("classes_names_codes", le.transform(classes_names))
print("Weights: ", list(class_weights.values()))
print("test_x classes_count: ", np.bincount(np.array(test_y)))



""" Dataset and loader """
entire_dataset = InfantDataset(data, classes)

train_dataset = InfantDataset(train_x, train_y)
valid_dataset = InfantDataset(valid_x, valid_y)
test_dataset = InfantDataset(test_x, test_y)


entire_ds_loader = DataLoader(
    dataset=entire_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
#         num_workers=2
)


train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
#         num_workers=2
)

valid_loader = DataLoader(
    dataset=valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
#         num_workers=2
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
#         num_workers=2
)


model = FeedForwardNet(inpNum = 1, shape = shape, classesNum = classesNum)
model = model#.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss(weight = torch.Tensor(list(class_weights.values())))
#     optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE,eps = 1e-5, weight_decay = 1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 5, verbose = True)
# scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=100, gamma=0.1)
#     print(model.summary())


#--------------------------------------------------------------------------------------
""" Training the model """
results = fit(model, train_loader, valid_loader, optimizer, loss_fn, EPOCHS, checkpoint_path, device)

#--------------------------------------------------------------------------------------
#model = results["model"]
model.load_state_dict(torch.load(checkpoint_path))


#-----------------------------------------------------------------------Results section
history = results["history"]
# plots, CM, testing
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.style.use("ggplot")


fig, ax = plt.subplots(1, 1, figsize=(5,5), tight_layout=True, frameon=True) 
plt.plot(history["acc"])
plt.plot(history["val_acc"])
plt.title("Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("epoch")
ax.set_xticks(np.arange(1, len(history["acc"])))

plt.legend(['train acc', 'loss acc'], loc='lower right')
#plt.show()
plt.savefig(f"{current_model}acc.png")


fig, ax = plt.subplots(1, 1, figsize=(5,5), tight_layout=True, frameon=True) 
plt.semilogy(history["loss"])
plt.semilogy(history["val_loss"])
plt.title("Loss")
plt.ylabel("Loss")
plt.xlabel("epoch")
plt.legend(['train loss', 'val loss'], loc='lower right')
ax.set_xticks(np.arange(1, len(history["acc"])))
#plt.show()
plt.savefig(f"{current_model}loss.png")


fig, ax = plt.subplots(1, 1, figsize=(5,5), tight_layout=True, frameon=True) 
plt.plot(history["lr"])
plt.title("Learning rate")
plt.ylabel("Learning rate")
plt.xlabel("epoch")
ax.set_xticks(np.arange(1, len(history["acc"])))
#plt.show()
plt.savefig(f"{current_model}lr.png")




print("----------------------------------------------------------Testing")
print("best_valid: ", results["best_valid"], " at: ", checkpoint_path)
test_loss, test_acc, y_test, y_pred_test, instance_acc_test = inference(model, test_loader, loss_fn, device)
ds_loss, ds_acc, y, y_pred_ds, instance_acc_ds = inference(model, entire_ds_loader, loss_fn, device)



precision_recall_fscore_support_test = mldl_uts.make_confusion_matrix(
     y = y_test,
     y_pred = y_pred_test,
     group_names = [],
     cmap = "jet",#"Greys",
     categories = classes_names,
     figsize = (9,7),
     title = "Confusion matrix",
     show = True, prefix = "");
#fig.savefig(f"{current_model}cm.png")






precision_recall_fscore_support_ds = mldl_uts.make_confusion_matrix(
     y = y,
     y_pred = y_pred_ds,
     group_names = [],
     cmap = "jet",#"Greys",
     categories = classes_names,
     figsize = (9,7),
     title = "Confusion matrix",
     show = True, prefix = "");
#fig.savefig(f"{current_model}cm.png")








df = {}


df["dataset"] = DATASET
x = DATASET.split("/")[1]
df["alg"] = "_".join(x.split("_")[1:-2])

df["imgSize"] = x.split("_")[-2]


df["epochs"] = "EPOCHS"
df["bs"] = BATCH_SIZE

df["lr"] = LEARNING_RATE


df["model"] = checkpoint_path
df["stop"] = results["epoch_best"]



#df["Tacc"] = ""
#df["Tloss"] = ""

#df["Vacc"] = ""
#df["Vloss"] = ""




df["loss_test"] = test_loss
df["loss_ds"] = ds_loss

df["acc_test"] = str(instance_acc_test)
df["acc_ds"] = str(instance_acc_ds)


#print("666666666666666: ", test_acc, precision_recall_fscore_support_test[0])




#df["precision_test"] = str(precision_recall_fscore_support_test[1][0])
#df["recall_test"] = str(precision_recall_fscore_support_test[1][1])
#df["f1_test"] = str(precision_recall_fscore_support_test[1][2])


#df["precision_ds"] = str(precision_recall_fscore_support_ds[1][0])
#df["recall_ds"] = str(precision_recall_fscore_support_ds[1][1])
#df["f1_ds"] = str(precision_recall_fscore_support_ds[1][2])



classes_labled = le.inverse_transform([0, 1, 2, 3, 4])


for i, (prec, rec, f1) in enumerate(zip(precision_recall_fscore_support_test[1][0], precision_recall_fscore_support_test[1][1], precision_recall_fscore_support_test[1][1])):
  df[classes_labled[i]+"_pre_test"] = prec
  df[classes_labled[i]+"_rec_test"] = rec
  df[classes_labled[i]+"_f1_test"] = f1


for i, (prec, rec, f1) in enumerate(zip(precision_recall_fscore_support_ds[1][0], precision_recall_fscore_support_ds[1][1], precision_recall_fscore_support_ds[1][1])):
  df[classes_labled[i]+"_pre_ds"] = prec
  df[classes_labled[i]+"_rec_ds"] = rec
  df[classes_labled[i]+"_f1_ds"] = f1




for k, v in instance_acc_ds.items():
  df[classes_labled[k]+"_acc_ds"] = v

df["acc_test"] = test_acc

for k, v in instance_acc_test.items():
  df[classes_labled[k]+"_acc_ds"] = v

df["acc_ds"] = ds_acc









from statistics import mean

df["pre_test"] = mean(precision_recall_fscore_support_test[1][0])
df["rec_test"] = mean(precision_recall_fscore_support_test[1][1])
df["f1_test"] = mean(precision_recall_fscore_support_test[1][2])


df["pre_ds"] = mean(precision_recall_fscore_support_ds[1][0])
df["rec_ds"] = mean(precision_recall_fscore_support_ds[1][1])
df["f1_ds"] = mean(precision_recall_fscore_support_ds[1][2])



df['tag'] = EXPERIMENT_TAG

#print(df)


df = pd.DataFrame.from_dict([df])
#print(df.columns)





cols = [
'dataset',
'imgSize',
'alg',
'epochs',
'bs',
'lr',

'stop',

'acc_test',
'loss_test',
'pre_test',
'rec_test',
'f1_test',


'acc_ds',
'loss_ds',
'pre_ds',
'rec_ds',
'f1_ds',


'belly_pain_f1_test',
'burping_f1_test',
'discomfort_f1_test',
'hungry_f1_test',
'tired_f1_test',


'belly_pain_pre_test',
'burping_pre_test',
'discomfort_pre_test',
'hungry_pre_test',
'tired_pre_test',

'belly_pain_f1_ds',
'burping_f1_ds',
'discomfort_f1_ds',
'hungry_f1_ds',
'tired_f1_ds',


'belly_pain_rec_test',
'burping_rec_test',
'discomfort_rec_test',
'hungry_rec_test',
'tired_rec_test',

'belly_pain_pre_ds',
'burping_pre_ds',
'discomfort_pre_ds',
'hungry_pre_ds',
'tired_pre_ds',

'belly_pain_rec_ds',
'burping_rec_ds',
'discomfort_rec_ds',
'hungry_rec_ds',
'tired_rec_ds',


'model',
'tag'
]


df = df[cols]

append2csv(f"Results/final_results.csv", df)





print("dataset: ", df["dataset"].tolist()[0])
print('stop: ', df['stop'].tolist()[0])
print('acc_test: ', df['acc_test'].tolist()[0])
print('loss_test: ', df['loss_test'].tolist()[0])
print('pre_test: ', df['pre_test'].tolist()[0])
print('rec_test: ', df['rec_test'].tolist()[0])
print('f1_test: ', df['f1_test'].tolist()[0])
print('acc_ds: ', df['acc_ds'].tolist()[0])
print('loss_ds: ', df['loss_ds'].tolist()[0])
print('pre_ds: ', df['pre_ds'].tolist()[0])
print('rec_ds: ', df['rec_ds'].tolist()[0])
print('f1_ds: ', df['f1_ds'].tolist()[0])



#print("\n\nInstance accuracy: ", instance_acc)
#print("\n\nInstance accuracy: ", instance_acc)

#print("\n\n", classification_report(y_test, y_pred), "\n\n")
#print("\n\n", classification_report(y, y_pred), "\n\n")

















