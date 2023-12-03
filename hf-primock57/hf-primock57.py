import datasets


class Primock57Dataset(datasets.GeneratorBasedBuilder):
    """
    HuggingFace compatible version of the Primock57 dataset
    https://github.com/babylonhealth/primock57
    """
    DEFAULT_CONFIG_NAME = "all"
    SPLITS = ["train", "test", "eval"]
    CSV_PATHS = {s: f"data/{s}.csv" for s in SPLITS}
    TAR_PATHS = {s: f"data/{s}.tar.gz" for s in SPLITS}

    def _info(self):
        return datasets.DatasetInfo(
            description=__class__.__doc__.strip(),
            features=datasets.Features(
                {
                    "path": datasets.Value("string"),
                    "audio": datasets.Audio(sampling_rate=16_000),
                    "transcription": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            # homepage=_HOMEPAGE,
            # license=_LICENSE,
            # citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        prompts_paths = dl_manager.download(self.CSV_PATHS)
        archive = dl_manager.download(self.TAR_PATHS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "prompts_path": prompts_paths["train"],
                    "audio_files": dl_manager.iter_archive(archive["train"]),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "prompts_path": prompts_paths["test"],
                    "audio_files": dl_manager.iter_archive(archive["test"]),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "prompts_path": prompts_paths["eval"],
                    "audio_files": dl_manager.iter_archive(archive["eval"]),
                },
            ),
        ]

    def _generate_examples(self, prompts_path, audio_files):
        """Yields examples as (key, example) tuples."""
        examples = {}
        with open(prompts_path, encoding="utf-8") as f:
            for row in f:
                audio_path, transcript = row.strip("\n").split(",")
                examples[audio_path] = {
                    "path": audio_path,
                    "transcription": transcript,
                }
        id_ = 0
        for path, f in audio_files:
            if path in examples:
                audio = {"path": path, "bytes": f.read()}
                yield id_, {**examples[path], "audio": audio}
                id_ += 1
