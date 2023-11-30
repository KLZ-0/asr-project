import re
from pathlib import Path
from typing import List

from datasets import Audio, Dataset
from textgrid import TextGrid, IntervalTier

from metadata import Interval


class Transcript:
    """
    Represents one transcript file, which consists of individual intervals
    """
    __path_re = re.compile(r"day(\d+)_consultation(\d+)_(\w+)")
    __path_re_keys = ["day", "consultation_n", "doctor"]

    _path: Path
    _intervals: List[Interval]
    __a = Audio(sampling_rate=16000)
    __cache = None

    day: int
    consultation_n: int
    is_doctor: bool

    def __init__(self, path: Path, day: int, consultation_n: int, doctor: bool):
        self._path = path
        self._audio = self.get_audio_from_path(path)
        self.day = day
        self.consultation_n = consultation_n
        self.is_doctor = doctor

    def save(self):
        for interv in self._intervals:
            interv.save()

    @property
    def fname(self):
        return self._path.name

    @property
    def audio(self):
        if self._audio is None:
            return None

        if self.__cache is None:
            self.__cache = Dataset.from_dict({
                "audio": [str(self._audio)]
            }).cast_column("audio", Audio())[0]["audio"]["array"]

        return self.__cache

    @property
    def intervals(self):
        return self._intervals

    @property
    def sid(self) -> str:
        return f"{self.day}:{self.consultation_n}:{int(self.is_doctor)}"

    @intervals.setter
    def intervals(self, it: IntervalTier):
        if not isinstance(it, IntervalTier):
            raise TypeError(f"Setting intervals with {it.__class__} (IntervalTier required)")

        self._intervals = [Interval.from_raw_interval(self, n + 1, i) for n, i in enumerate(it)]

    @classmethod
    def from_file(cls, path: Path) -> "Transcript":
        tmp = cls(path=path, **cls.decode_path_name(path.stem))
        tmp.intervals = TextGrid.fromFile(path)[0]

        return tmp

    @classmethod
    def from_dict(cls) -> "Transcript":
        # For future use when loading from a pre-processed file
        raise NotImplementedError()

    @classmethod
    def decode_path_name(cls, path_name: str) -> dict:
        tmp = dict(zip(cls.__path_re_keys, re.match(cls.__path_re, path_name).groups()))
        tmp["day"] = int(tmp["day"])
        tmp["consultation_n"] = int(tmp["consultation_n"])
        tmp["doctor"] = True if tmp["doctor"] == "doctor" else False
        return tmp

    @staticmethod
    def get_audio_path_from_path(path: Path) -> Path:
        audio_fname = path.parent.parent / "audio" / path.name
        return audio_fname.with_suffix(".wav")

    @classmethod
    def get_audio_from_path(cls, path: Path):
        audio_path = cls.get_audio_path_from_path(path)
        if not audio_path.exists():
            return None

        return audio_path

    def __str__(self):
        return f"Transcript(day={self.day}, n={self.consultation_n}, doc={self.is_doctor})"
