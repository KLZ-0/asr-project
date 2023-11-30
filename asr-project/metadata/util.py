import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, List

from rich.progress import track

from metadata import Interval


def generate(dct: Dict[str, List[Interval]], out_dir: Path = Path(".")):
    out_dir.mkdir(parents=True, exist_ok=True)
    DATA_DIR = out_dir / "data"
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
