from lhotse.recipes import download_librispeech, prepare_librispeech
from lhotse import CutSet, Fbank

#download_librispeech(dataset_parts='mini_librispeech', target_dir='downloads/')
libri = prepare_librispeech(corpus_dir='downloads/LibriSpeech', output_dir='data/mini_librispeech')

# CutSet is the workhorse of Lhotse, allowing for flexible data manipulation.
# We create 5-second cuts by traversing SWBD recordings in windows.
# No audio data is actually loaded into memory or stored to disk at this point.  

cuts_train = CutSet.from_manifests(**libri['train-clean-5'])
cuts_dev = CutSet.from_manifests(**libri['dev-clean-2'])

# We compute the log-Mel filter energies and store them on disk;
# Then, we pad the cuts to 5 seconds to ensure all cuts are of equal length,
# as the last window in each recording might have a shorter duration.
# The padding will be performed once the features are loaded into memory.

cuts_train = cuts_train.compute_and_store_features(
    extractor=Fbank(),
    storage_path='data/mini_librispeech/feats_train-clean-5',
    num_jobs=1
).pad(duration=5.0)

cuts_train.to_json("data/mini_librispeech/feats_train-clean-5.json.gz")

cuts_dev = cuts_dev.compute_and_store_features(
    extractor=Fbank(),
    storage_path='data/mini_librispeech/feats_dev-clean-2',
    num_jobs=1
).pad(duration=5.0)

cuts_dev.to_json("data/mini_librispeech/feats_dev-clean-2.json.gz")