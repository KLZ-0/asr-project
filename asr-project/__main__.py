import random
from pathlib import Path

from metadata import DataSet

random.seed(0)


def main():
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

    OUT_DIR = Path(".")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for key, intrvls in dct.items():
        with (OUT_DIR / key).open("w") as f:
            for it in intrvls:
                f.write(f"{it.sid}\t{it.text}\n")

        with (OUT_DIR / key).with_suffix(".s").open("w") as f:
            for it in intrvls:
                f.write(f"<s> {it.text} </s>\n")


if __name__ == '__main__':
    main()
