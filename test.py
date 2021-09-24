import glob
import sox
import json
import numpy as np
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from lhotse import LilcomFilesWriter

from lhotse.features import Fbank, FeatureSetBuilder
from lhotse.cut import CutSet, SupervisionSet
from lhotse.dataset.sampling import SingleCutSampler
from lhotse.dataset.vad import VadDataset

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from preprocessing import prepare_vad_dataset

root_dir = Path('evaluation/data')
corpus_dir = root_dir / 'vad_data/'
output_dir = root_dir / 'vad_data_nb/'

vad_manifests = prepare_vad_dataset.prepare_vad_dataset(corpus_dir, output_dir)

cuts = CutSet.from_file(output_dir / 'cuts.json.gz')

print(int(len(vad_manifests["supervisions"])*0.8))
print(len(vad_manifests["supervisions"]))

train_ids = vad_manifests["supervisions"][:int(len(vad_manifests["supervisions"])*0.8)]
dev_ids = vad_manifests["supervisions"][len(train_ids):int((len(vad_manifests["supervisions"])-len(train_ids))/2)]
eval_ids = vad_manifests["supervisions"][len(train_ids)+len(dev_ids):]

sys.exit()

cuts.describe()

# Shuffle data but keep seed fixed, split into 80/10/10
cuts_train, cuts_dev_eval = train_test_split(cuts, train_size=0.8, random_state=0)
cuts_dev, cuts_eval = train_test_split(cuts_dev_eval, train_size=0.5, random_state=0)

cuts_train = CutSet(cuts_train)
cuts_dev = CutSet(cuts_dev)
cuts_eval = CutSet(cuts_eval)

train_sampler = SingleCutSampler(cuts_train.cut_into_windows(5.0, keep_excessive_supervisions=True), shuffle=False)
dev_sampler = SingleCutSampler(cuts_dev.cut_into_windows(5.0, keep_excessive_supervisions=True), shuffle=False)
eval_sampler = SingleCutSampler(cuts_eval.cut_into_windows(5.0, keep_excessive_supervisions=True), shuffle=False)

train_dloader = DataLoader(vad_dataset, sampler=train_sampler, batch_size=None, num_workers=1)
dev_dloader = DataLoader(vad_dataset, sampler=dev_sampler, batch_size=None, num_workers=1)
eval_dloader = DataLoader(vad_dataset, sampler=eval_sampler, batch_size=None, num_workers=1)
