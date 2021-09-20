import glob
import sox
import json
import numpy as np
import os
from pathlib import Path
from preprocessing import prepare_vad_dataset

root_dir = Path('evaluation/data')
corpus_dir = root_dir / 'vad_data/'
output_dir = root_dir / 'vad_data_nb/'

vad_manifests = prepare_vad_dataset.prepare_vad_dataset(corpus_dir, output_dir)
