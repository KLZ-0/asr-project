from pathlib import Path
from typing import List

from .transcript import Transcript


class DataSet:
    """
    Represents the entire dataset, consisting of individual transcripts
    """
    _transcripts = []

    def __init__(self, path: Path):
        self._transcripts_path = path / "transcripts"
        self._audio_path = path / "audio"

        for p in self._transcripts_path.glob("*.TextGrid"):
            self._transcripts.append(Transcript.from_file(p))
            # tr = Transcript.from_file(p)
            # tr.save()
            # del tr

        # audio_dataset = Dataset.from_dict({
        #     "audio": [str(p) for p in self._audio_path.glob("*.wav")]
        # }).cast_column("audio", Audio())
        # print(audio_dataset[0])

    @property
    def transcripts(self) -> List[Transcript]:
        return self._transcripts
