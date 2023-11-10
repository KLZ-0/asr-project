from textgrid import Interval as ForeignInterval


class Interval:
    """
    Represents one interval in a transcript file
    """

    _transcript: object
    n: int
    start_time: float
    end_time: float
    text: str

    def __str__(self):
        return f"Interval(tr_sid={self._transcript.sid}, n={self.n}, '{self.text}')"

    @classmethod
    def from_raw_interval(cls, _transcript: object, n: int, interval: ForeignInterval):
        tmp = cls()
        tmp._transcript = _transcript
        tmp.n = n
        tmp.start_time = interval.minTime
        tmp.end_time = interval.maxTime
        tmp.text = interval.mark
        return tmp

    @property
    def sid(self) -> str:
        tr = self._transcript
        # return f"{tr.sid}:{self.n}"
        return f"{tr.sid}:{self.start_time}:{self.end_time}"
