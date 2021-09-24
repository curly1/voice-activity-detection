from pathlib import Path

from lhotse import LilcomFilesWriter

from lhotse.features import Fbank, FbankConfig
from lhotse.cut import CutSet

import prepare_vad_dataset

def main():

    # Set paths
    root_dir = Path('evaluation/data')
    corpus_dir = root_dir / 'vad_data/'
    output_dir = root_dir / 'vad_data_nb/'

    # Other settings
    num_mel_bins = 80

    # Prepare VAD dataset 
    vad_manifests = prepare_vad_dataset.prepare_vad_dataset(corpus_dir, output_dir)

    # Compute and store features
    cuts = CutSet.from_manifests(
        recordings=vad_manifests['recordings'],
        supervisions=vad_manifests['supervisions']
    ).compute_and_store_features(
        extractor=Fbank(config=FbankConfig(num_mel_bins=num_mel_bins)),
        storage_path=f'{output_dir}/feats_{str(num_mel_bins)}',
        storage_type=LilcomFilesWriter,
    )

    # Store cuts
    cuts.to_json(output_dir / f'cuts_{str(num_mel_bins)}.json.gz')


if __name__ == '__main__':
    main()