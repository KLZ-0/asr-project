# General imports
import torch
import json

from transformers import \
    AutoProcessor, Wav2Vec2Processor, Wav2Vec2ProcessorWithLM, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer
from pyctcdecode import build_ctcdecoder
from pathlib import Path
from evaluate import load

from datasets import load_dataset
sample_dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
dataset = load_dataset("hf-primock57", split="train")



path_to_models = Path("models")



pre_trained_model = "facebook/wav2vec2-base-960h" # "hf-test/xls-r-300m-sv" # Is an LM-less
processor = AutoProcessor.from_pretrained(pre_trained_model)



tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(pre_trained_model)
sorted_vocab_dict = {k.lower(): v for k, v in sorted(tokenizer.get_vocab().items(), key=lambda item: item[1])}

with open("vocab.json", "w") as outfile:
    json.dump(sorted_vocab_dict, outfile)
tokenizer = Wav2Vec2CTCTokenizer(vocab_file="vocab.json")



current_dir = Path().cwd()
context_2_gram = path_to_models / "2gram_context_lm.arpa"
background_2_gram = path_to_models / "2gram_background_lm.arpa"
adapted_2_gram = path_to_models / "2gram_log_adapted_lm.arpa"

# Create 2-gram based decoder
decoder_context_2g = build_ctcdecoder(
    labels=list(tokenizer.get_vocab().keys()),
    kenlm_model_path=str(context_2_gram),
)

decoder_background_2g = build_ctcdecoder(
    labels=list(tokenizer.get_vocab().keys()),
    kenlm_model_path=str(background_2_gram),
)

decoder_adapted_2g = build_ctcdecoder(
    labels=list(tokenizer.get_vocab().keys()),
    kenlm_model_path=str(adapted_2_gram),
)



# Create processor without LM for comparison
processor_no_lm = Wav2Vec2Processor.from_pretrained(pre_trained_model)

# Create processor based on 2-gram decoder
processor_context_2g = Wav2Vec2ProcessorWithLM(
    feature_extractor=processor.feature_extractor,
    tokenizer=tokenizer,
    decoder=decoder_context_2g
)

processor_background_2g = Wav2Vec2ProcessorWithLM(
    feature_extractor=processor.feature_extractor,
    tokenizer=tokenizer,
    decoder=decoder_background_2g
)

processor_adapted_2g = Wav2Vec2ProcessorWithLM(
    feature_extractor=processor.feature_extractor,
    tokenizer=tokenizer,
    decoder=decoder_adapted_2g
)


tmp_i = 0
while (audio_sample := dataset[tmp_i])["transcription"] == "":
    tmp_i += 1

# audio_sample["text"] = audio_sample["text"].lower()

# Generating input vectors from each processor
inputs_no_lm = processor_no_lm(audio_sample["audio"]["array"], sampling_rate=audio_sample["audio"]["sampling_rate"], return_tensors="pt")

inputs_context_2g = processor_context_2g(audio_sample["audio"]["array"], sampling_rate=audio_sample["audio"]["sampling_rate"], return_tensors="pt")
inputs_background_2g = processor_background_2g(audio_sample["audio"]["array"], sampling_rate=audio_sample["audio"]["sampling_rate"], return_tensors="pt")
inputs_adapted_2g = processor_adapted_2g(audio_sample["audio"]["array"], sampling_rate=audio_sample["audio"]["sampling_rate"], return_tensors="pt")



model = Wav2Vec2ForCTC.from_pretrained(pre_trained_model)

def get_logit(inp):
    with torch.no_grad():
        return model(**inp).logits.numpy()
# with torch.no_grad():
  # logits_no_lm = model(**inputs_no_lm).logits

tmp = get_logit(inputs_context_2g)
processor_context_2g.batch_decode(tmp).text



model = Wav2Vec2ForCTC.from_pretrained(pre_trained_model)

def get_logit(input):
    with torch.no_grad():
        return model(**input).logits.numpy()
with torch.no_grad():
  logits_no_lm = model(**inputs_no_lm).logits

logits_context_2g = get_logit(inputs_context_2g)
logits_background_2g = get_logit(inputs_background_2g)
logits_adapted_2g = get_logit(inputs_adapted_2g)



predicted_ids = torch.argmax(logits_no_lm, dim=-1)
transcription_no_lm = processor_no_lm.batch_decode(predicted_ids)

transcription_context_2g = processor_context_2g.batch_decode(logits_context_2g).text
transcription_background_2g = processor_background_2g.batch_decode(logits_background_2g).text
transcription_adapted_2g = processor_adapted_2g.batch_decode(logits_adapted_2g).text

reference = audio_sample["transcription"].lower()
no_lm = transcription_no_lm[0].lower()
context = transcription_context_2g[0].lower()
background = transcription_background_2g[0].lower()
combined = transcription_adapted_2g[0].lower()

print("Original: ", reference)
print("Without LM: ", no_lm)
print("Context 2G: ", context)
print("Background 2G: ", background)
print("Combined 2G: ", combined)

predictions = [no_lm, context, background, combined]

wer_metric = load("wer")
wer = [wer_metric.compute(references=[reference], predictions=[pred]) for pred in predictions]
print("=== WER ===")
print("Without LM: ", wer[0])
print("Context 2G: ", wer[1])
print("Background 2G: ", wer[2])
print("Combined 2G: ", wer[3])
