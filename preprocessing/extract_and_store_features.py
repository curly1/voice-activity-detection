from pathlib import Path
import argparse

from lhotse import LilcomFilesWriter

from lhotse.features import Fbank, FbankConfig
from lhotse.augmentation import SoxEffectTransform, RandomValue, pitch, reverb, speed
from lhotse.cut import CutSet

import prepare_vad_dataset

def main():

    parser = argparse.ArgumentParser(description="Extract and store features.")
    parser.add_argument(
        "--data-dir", type=str, default=Path('evaluation/data/vad_data')
    )
    parser.add_argument(
        "--data-augment", type=bool, default=False
    )
    args = parser.parse_args()

    corpus_dir = args.data_dir
    output_dir = corpus_dir + "_nb"
    use_data_augmentation = args.data_augment

    # Other settings
    num_mel_bins = 80
    
    # Prepare VAD dataset 
    vad_manifests = prepare_vad_dataset.prepare_vad_dataset(
                    corpus_dir, output_dir)

    # Compute and store features
    cuts = CutSet.from_manifests(
        recordings=vad_manifests['recordings'],
        supervisions=vad_manifests['supervisions'])

    # Speed and volume perturbation
    if use_data_augmentation:
        cuts = cuts + cuts.perturb_speed(0.9) + cuts.perturb_speed(1.1) + \
            cuts.perturb_volume(1.8) + cuts.perturb_volume(0.8)

    # Reverberation
    augment_fn = SoxEffectTransform(effects=[
        ['reverb', 50, 50, RandomValue(0, 100)],
        ['remix', '-'],  # Merge all channels (reverb changes mono to stereo)
    ]) if use_data_augmentation else None

    suffix = "_data_augment" if use_data_augmentation else ""

    cuts = cuts.compute_and_store_features(
        extractor=Fbank(config=FbankConfig(num_mel_bins=num_mel_bins)),
        storage_path=f'{output_dir}/feats_{str(num_mel_bins)}{suffix}',
        augment_fn=augment_fn,
        storage_type=LilcomFilesWriter,
    )

    # Store cuts
    cuts.to_json(output_dir / f'cuts_{str(num_mel_bins)}{suffix}.json.gz')


if __name__ == '__main__':
    main()