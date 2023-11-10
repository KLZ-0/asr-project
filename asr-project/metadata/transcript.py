import re
from pathlib import Path
from typing import List

from textgrid import TextGrid, IntervalTier

from metadata import Interval


class Transcript:
    """
    Represents one transcript file, which consists of individual intervals
    """
    __path_re = re.compile(r"day(\d+)_consultation(\d+)_(\w+)")
    __path_re_keys = ["day", "consultation_n", "doctor"]

    # NOTE: some marks contain only tags such as "<INAUDIBLE_SPEECH/>", not sure if we need to filter those
    __valid_text_contains_re = re.compile(r"[a-z]")

    _fname: str
    _intervals: List[Interval]

    day: int
    consultation_n: int
    is_doctor: bool

    def __init__(self, fname: str, day: int, consultation_n: int, doctor: bool):
        self._fname = fname
        self.day = int(day)
        self.consultation_n = int(consultation_n)
        self.is_doctor = doctor

    @property
    def fname(self):
        return self._fname

    @property
    def intervals(self):
        return self._intervals

    @property
    def sid(self) -> str:
        return f"{self.day}:{self.consultation_n}:{int(self.is_doctor)}"

    @classmethod
    def is_text_ok(cls, text: str):
        # if not text:
        #     return False

        if not re.search(cls.__valid_text_contains_re, text):
            return False

        return True

    @intervals.setter
    def intervals(self, it: IntervalTier):
        self._intervals = [Interval.from_raw_interval(self, n + 1, i)
                           for n, i in enumerate(it) if self.is_text_ok(i.mark)]

    @classmethod
    def from_file(cls, path: Path):
        tmp = cls(fname=path.name, **cls.decode_path_name(path.stem))
        tmp.intervals = TextGrid.fromFile(path)[0]

        return tmp

    @classmethod
    def from_dict(cls):
        # For future use when loading from a pre-processed file
        raise NotImplementedError()

    @classmethod
    def decode_path_name(cls, path_name: str) -> dict:
        tmp = dict(zip(cls.__path_re_keys, re.match(cls.__path_re, path_name).groups()))
        tmp["doctor"] = True if tmp["doctor"] == "doctor" else False
        return tmp

    def __str__(self):
        return f"Transcript(day={self.day}, n={self.consultation_n}, doc={self.is_doctor})"
