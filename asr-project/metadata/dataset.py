from pathlib import Path
from typing import List

from metadata.transcript import Transcript


class DataSet:
    """
    Represents the entire dataset, consisting of individual transcripts
    """
    _transcripts = []

    def __init__(self, path: Path):
        self._transcripts_path = path / "transcripts"
        for p in self._transcripts_path.glob("*.TextGrid"):
            self._transcripts.append(Transcript.from_file(p))

    @property
    def transcripts(self) -> List[Transcript]:
        return self._transcripts
