import glob
import logging

from pathlib import Path
from lhotse.utils import Pathlike
from typing import Dict, List, Optional, Sequence, Tuple, Union
from lhotse.audio import Recording, RecordingSet
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.supervision import SupervisionSegment, SupervisionSet

def prepare_vad_dataset(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:

    """
    Returns the manifests which consist of the Recordings and Supervisions.
    When all the manifests are available in the ``output_dir``, it will simply read and return them.
    
    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'audio' and 'supervisions'.
    """
    
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f'No such directory: {corpus_dir}'
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Generate a mapping: utt_id -> (audio_path, audio_info, text)
    recordings = []
    supervisions = []
    for json_file in glob.glob(corpus_dir + "*.json"):
        recording_id = json_file.split("/")[-1].strip(".json")
        audio_path = f'{recording_id}.wav'
        if not audio_path.is_file():
            logging.warning(f'No such file: {audio_path}')
            continue    
        recording = Recording.from_file(audio_path)
        
        segment = SupervisionSegment(
            id=recording_id,
            recording_id=recording_id,
            start=float(start),
            duration=round(float(end) - float(start), ndigits=8),
            language='English',
            )
        recordings.append(recording)
        supervisions.append(segment)

    recording_set = RecordingSet.from_recordings(recordings)
    supervision_set = SupervisionSet.from_segments(supervisions)

    validate_recordings_and_supervisions(recording_set, supervision_set)

    if output_dir is not None:
        supervision_set.to_json(output_dir / 'supervisions.json')
        recording_set.to_json(output_dir / 'recordings.json')

    return {'recordings': recording_set, 'supervisions': supervision_set}