import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset
from rich.progress import track

from . import Interval


def generate(dct: Dict[str, List[Interval]], out_dir: Path = Path(".")):
    out_dir.mkdir(parents=True, exist_ok=True)
    DATA_DIR = out_dir / "hf-primock57" / "data"
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tars = {k: tarfile.open((DATA_DIR / k).with_suffix(".tar.gz"), "w:gz") for k, i in dct.items()}
    for key, intrvls in dct.items():
        with tempfile.TemporaryDirectory() as tmp_dir:
            md = (DATA_DIR / f"{key}.csv").open("w")
            md.write("file_name,transcription\n")
            for it in track(intrvls, description=f"Processing {key}:"):
                name, text = it.save(Path(tmp_dir))
                if name == Path():
                    continue
                md.write(f"{name.name},{text}\n")
                tars[key].add(name, arcname=name.name)

            md.close()
            # tars[key].add((Path(tmp_dir) / "metadata.csv"), arcname="metadata.csv")
    [v.close() for v in tars.values()]


def sentences(dct: Dict[str, List[Interval]], out_dir: Path = Path(".")):
    out_dir /= "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, intrvls in dct.items():
        with (out_dir / key).open("w") as f:
            for it in intrvls:
                f.write(f"{it.sid}\t{it.text}\n")

        with (out_dir / key).with_suffix(".s").open("w") as f:
            for it in intrvls:
                f.write(f"<s> {it.text} </s>\n")


def test(out_dir: Path = Path(".")):
    DATA_DIR = out_dir / "hf-primock57" / "data"
    if not DATA_DIR.exists() or not DATA_DIR.is_dir():
        raise NotADirectoryError(f"Data directory '{DATA_DIR}' is not a directory!")

    dataset = load_dataset("hf-primock57")
    print(dataset)
    print(dataset["train"][2])
