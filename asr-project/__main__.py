import random
import sys
from pathlib import Path

import metadata
from metadata import DataSet

random.seed(0)

HELP = """
=== Dataset generator ===
Possible arguments:
    data - generate huggingface dataset
    sent - generate clean sentences for the n-gram model
"""


def main():
    if len(sys.argv) < 2:
        print(HELP.strip())
        exit()

    if sys.argv[1] == "test":
        metadata.test()
        exit()

    ds = DataSet(Path("./primock57"))  # give the primock57 path
    intervals = []
    for tr in ds.transcripts:
        # print(tr, len(tr.intervals))
        intervals.extend(tr.intervals)

    random.shuffle(intervals)
    n_total = len(intervals)
    n_test = n_eval = int(n_total * 0.1)
    n_train = n_total - n_test - n_eval

    print(f"{n_train = }")
    print(f"{n_test = }")
    print(f"{n_eval = }")
    print(f"{n_total = }")

    dct = {
        "train": intervals[:n_train],
        "test": intervals[n_train:n_train + n_test],
        "eval": intervals[n_train + n_test:]
    }

    if sys.argv[1] == "data":
        metadata.generate(dct)
    elif sys.argv[1] == "sent":
        metadata.sentences(dct)
    else:
        raise ValueError(f"Unknown argument '{sys.argv[1]}'")


if __name__ == '__main__':
    main()
