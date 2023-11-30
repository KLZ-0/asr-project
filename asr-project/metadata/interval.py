import re
from typing import TYPE_CHECKING

from textgrid import Interval as ForeignInterval

if TYPE_CHECKING:
    from metadata.transcript import Transcript


class Interval:
    """
    Represents one interval in a transcript file
    """

    _transcript: "Transcript"
    n: int
    start_time: float
    end_time: float
    text: str

    __retmp = re.compile(r"(</?\w+/?>|[^a-z ])")

    def __str__(self):
        return f"Interval(tr_sid={self._transcript.sid}, n={self.n}, '{self.text}')"

    @classmethod
    def from_raw_interval(cls, transcript: "Transcript", n: int, interval: ForeignInterval) -> "Interval":
        tmp = cls()
        tmp._transcript = transcript
        tmp.n = n
        tmp.start_time = interval.minTime
        tmp.end_time = interval.maxTime
        tmp.text = re.sub(cls.__retmp, "", interval.mark.lower())
        if transcript.audio:
            tmp.audio = transcript.audio[0]["audio"]["array"][int(tmp.start_time * 16000):int(tmp.end_time * 16000)]
        return tmp

    @property
    def sid(self) -> str:
        tr = self._transcript
        # return f"{tr.sid}:{self.n}"
        return f"{tr.sid}:{self.start_time}:{self.end_time}"
