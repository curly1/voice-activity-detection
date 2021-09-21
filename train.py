import glob
import sox
import json
import numpy as np
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from lhotse import LilcomFilesWriter

from lhotse.features import Fbank, FeatureSetBuilder
from lhotse.cut import CutSet, SupervisionSet
from lhotse.dataset.sampling import SingleCutSampler
from lhotse.dataset.vad import VadDataset

from torch.utils.data import DataLoader
import torch.nn as nn

from preprocessing import prepare_vad_dataset
from models.binary_dnn import binaryClassification
from models.dnn import DNN

root_dir = Path('evaluation/data')
corpus_dir = root_dir / 'vad_data/'
output_dir = root_dir / 'vad_data_nb/'

cuts = CutSet.from_json(output_dir / 'cuts.json.gz')
#cuts.describe()

vad_manifests = prepare_vad_dataset.prepare_vad_dataset(corpus_dir, output_dir)

train_ratio = 0.8

num_total = len(vad_manifests["supervisions"])
stop_train_idx = int(np.floor(num_total * train_ratio))
stop_dev_idx = int((num_total - stop_train_idx) // 2 + stop_train_idx)

train_ids, dev_ids, eval_ids = [], [], []
counter = 0
for sup_seg in vad_manifests["supervisions"]:
    id = sup_seg.to_dict()["id"]
    if counter < stop_train_idx:
        train_ids.append(id)
    elif counter < stop_dev_idx:
        dev_ids.append(id)
    else:
        eval_ids.append(id)
    counter += 1

assert train_ids[-1] != dev_ids[0]
assert dev_ids[-1] != eval_ids[0]

cuts_train = cuts.subset(supervision_ids=train_ids)
cuts_dev = cuts.subset(supervision_ids=dev_ids)
cuts_eval = cuts.subset(supervision_ids=eval_ids)

#cuts_train.describe()
#cuts_dev.describe()
#cuts_eval.describe()

vad_dataset = VadDataset()

train_sampler = SingleCutSampler(cuts_train.cut_into_windows(5.0, keep_excessive_supervisions=True), shuffle=False)
dev_sampler = SingleCutSampler(cuts_dev.cut_into_windows(5.0, keep_excessive_supervisions=True), shuffle=False)
eval_sampler = SingleCutSampler(cuts_eval.cut_into_windows(5.0, keep_excessive_supervisions=True), shuffle=False)

train_dloader = DataLoader(vad_dataset, sampler=train_sampler, batch_size=None, num_workers=0)
dev_dloader = DataLoader(vad_dataset, sampler=dev_sampler, batch_size=None, num_workers=0)
eval_dloader = DataLoader(vad_dataset, sampler=eval_sampler, batch_size=None, num_workers=0)

#cut_ids = next(iter(dev_sampler))
#sample = vad_dataset[cut_ids]

input_size = 40

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#model = DNN(input_size=input_size, hidden_size=500, num_classes=2).to(device)
model = binaryClassification().to(device)

criterion = nn.BCEWithLogitsLoss()

optim = torch.optim.Adam(model.parameters())

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

train_acc = []
valid_acc = []
for epoch in range(10):
  # training
    acc = []
    model.train()
    train_dloader.sampler.set_epoch(epoch)

    for batch_idx, data in enumerate(train_dloader):

        inputs = data["inputs"].reshape(-1,input_size)
        targets = data["is_voice"].reshape(-1,1).view(-1)

        out = model(inputs.to(device))

        loss = criterion(out, targets.unsqueeze(1).to(device))
        model_acc = binary_acc(out, targets.unsqueeze(1).to(device))

        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx} => loss {loss}')
        optim.zero_grad()
        loss.backward()
        acc.append(model_acc)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optim.step()
        t_r = loss.item()
    train_acc.append(np.mean(acc))
    # validation
    acc = []
    model.eval()
    for data in dev_dloader:
        inputs = data["inputs"].reshape(-1,input_size)
        targets = data["is_voice"].reshape(-1,1).view(-1)
        out = model(inputs.to(device))
        model_acc = binary_acc(out, targets.unsqueeze(1).to(device))
        acc.append(model_acc)
    valid_acc.append(np.mean(acc))
    print(f"epoch: {epoch}, train acc: {train_acc[-1]:.3f}, dev acc: {valid_acc[-1]:.3f}, loss:{t_r:.3f}")
