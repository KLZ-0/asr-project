import random
import shutil
import tarfile
import tempfile
from pathlib import Path

from rich.progress import track

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

    lines = []
    OUT_DIR = Path(".")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR = OUT_DIR / "data"
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tars = {k: tarfile.open((DATA_DIR / k).with_suffix(".tar.gz"), "w:gz") for k, i in dct.items()}
    md = (DATA_DIR / "metadata.csv").open("w")
    for key, intrvls in dct.items():
        with tempfile.TemporaryDirectory() as tmp_dir:
            for it in track(intrvls, description=f"Processing {key}:"):
                name, text = it.save(Path(tmp_dir))
                if not text:
                    continue
                md.write(f"{name.name},{text}\n")
                tars[key].add(name, arcname=name.name)

    md.close()
    [v.close() for v in tars.values()]
    # for key, intrvls in dct.items():
    #     with (OUT_DIR / key).open("w") as f:
    #         for it in intrvls:
    #             f.write(f"{it.sid}\t{it.text}\n")
    #
    #     with (OUT_DIR / key).with_suffix(".s").open("w") as f:
    #         for it in intrvls:
    #             f.write(f"<s> {it.text} </s>\n")

if __name__ == '__main__':
    main()
