
# To run this file you might need the following modules (pip install):
#    datasets transformers soundfile evaluate pyctcdecode pypi-kenlm jiwer https://github.com/kpu/kenlm/archive/master.zip
    
# General imports
import torch
import json
import warnings

from pathlib import Path

from datasets import load_dataset
from transformers import AutoProcessor, Wav2Vec2ProcessorWithLM, \
                            Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer
from pyctcdecode import build_ctcdecoder
from evaluate import load
from rich.progress import track

class ModelEvaluator:
    """
    Build and evaluate multiple n-gram models
    """
    def __init__(self, N: list, model_names: list, n_samples: int=-1, models_dir: Path=""):
        if (type(N) == int): N = [N]
        if (type(model_names) == int): model_names = [model_names]
        
        self.N = N
        self.model_names = model_names
        self.model_dir = Path(models_dir)
        
        self.dataset = load_dataset("hf-primock57", split="test")
        
        self.pre_trained_model = "facebook/wav2vec2-base-960h"
        print("Loading pre-trained model: ", self.pre_trained_model)
        self.base_processor = AutoProcessor.from_pretrained(self.pre_trained_model)
        self.accoustic_model = Wav2Vec2ForCTC.from_pretrained(self.pre_trained_model)
        self.create_tokenizer()
        
        # TODO: change this:
        print("Extract audio sample ...", end=" ")
        if(n_samples == -1): 
            self.num_samples = self.dataset.num_rows
        else:
            self.num_samples = min(self.dataset.num_rows, n_samples)
            
        self.audio_sample = self.dataset[:self.num_samples]
        print("[DONE]")
        

    def create_tokenizer(self):
        """
        Create a tokenizer making sure that the specified alphabet is in
        lower case. This is done by taking a tokenizer with a regular
        alphabet, converting it to lower case and saving it to:
            vocab.json
        A new tokenizer is then created with vocab_file="vocab.json"
        """
        print("Creating tokenizer...", end=" ")
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(self.pre_trained_model)
        sorted_vocab_dict = {k.lower(): v for k, v in sorted(tokenizer.get_vocab().items(), key=lambda item: item[1])}
        
        with open("vocab.json", "w") as outfile:
            json.dump(sorted_vocab_dict, outfile)
        self.tokenizer = Wav2Vec2CTCTokenizer(vocab_file="vocab.json")
        print("[DONE]")
        

    def generate_transcripts(self):
        if not hasattr(self, 'tokenizer'):
            self.create_tokenizer()
        model_paths = {}
        decoders = {}
        processors = {}
        inputs = {}
        logits = {}
        transcripts = {}
        
        for name in self.model_names:
            model_paths[name] = {}
            decoders[name] = {}
            processors[name] = {}
            inputs[name] = {}
            logits[name] = {}
            transcripts[name] = {}
            
            for n in self.N:
                print(f"Processing {name} {n}-gram...")
                # model path
                model_paths[name][n] = Path().cwd() / "models" / str(str(n) + "gram_" + name + "_lm.arpa")
                # Decoder
                print("\tGenerating decoder")
                decoders[name][n] = build_ctcdecoder(
                    labels=list(self.tokenizer.get_vocab().keys()),
                    kenlm_model_path=str(model_paths[name][n]))
                # Processor
                print("\tGenerating processor")
                processors[name][n] = Wav2Vec2ProcessorWithLM(
                    feature_extractor=self.base_processor.feature_extractor,
                    tokenizer=self.tokenizer,
                    decoder=decoders[name][n]
                )
                # Compute inputs for acoustic model 
                print("\tComputing Wav2Vec2 embeddings")
                inputs[name][n] = []
                for i in range(self.num_samples):
                    inputs[name][n].append(processors[name][n](
                        self.audio_sample["audio"][i]["array"], 
                        sampling_rate=self.audio_sample["audio"][i]["sampling_rate"], 
                        return_tensors="pt"))
                # Compute logits
                logits[name][n] = []
                #print("\tFeeding embeddings to acoustic models 0", end="")
                with torch.no_grad():
                    for i in track(range(self.num_samples), description="Feeding embeddings"):
                        logits[name][n].append(self.accoustic_model(**inputs[name][n][i]).logits.numpy())
                print()
                # Compute final transcripts
                print("\tDecoding output")
                transcripts[name][n] = processors[name][n].batch_decode(logits[name][n]).text
        
        self.model_paths = model_paths
        self.decoders = decoders
        self.processors = processors 
        self.inputs = inputs
        self.logits = logits
        self.transcripts = transcripts 
    
    def print_transcripts(self, nb=-1):
        if nb == -1: nb = self.num_samples
        print("----------")
        for i in range(min(nb, self.num_samples)):
            print("Original: ", self.audio_sample["transcription"][i])
            for name, val in self.transcripts.items():
                for n, v in val.items():
                    print(f"{name} {n}-gram: ", v[i])
            print()
        print("----------")
    
    def compute_wer(self):
        wer_metric = load("wer")
        wer = {}
        for name in self.model_names:
            wer[name] = {}
            for n in self.N:
                reference = self.audio_sample["transcription"]
                predictions = self.transcripts[name][n]
                wer[name][n] = [wer_metric.compute(references=reference, predictions=predictions)]
        print(wer)
    
    
    def __compute_wer(reference, prediction):
        wer_metric = load("wer")
        # TODO

    
if __name__ == "__main__":
    model_names = ["context", "background", "log_adapted"] # "background"
    N = [2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        evaluator = ModelEvaluator(N, model_names, n_samples=50)
        evaluator.create_tokenizer()
        evaluator.generate_transcripts()
        # evaluator.print_transcripts(5)
        evaluator.compute_wer()
        
    
    
    
    
    