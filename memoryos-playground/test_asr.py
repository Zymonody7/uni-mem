from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import librosa

model_dir = "/root/models/whisper-large-v3"

processor = WhisperProcessor.from_pretrained(model_dir)
model = WhisperForConditionalGeneration.from_pretrained(model_dir).to("cuda")

audio, sr = librosa.load("/root/repo/memvideo/memoryos-playground/memdemo/videorag-workdir/_cache/test_video/1764843452053-87-2610-2640.mp3", sr=16000)
input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to("cuda")

forced_decoder_ids = processor.get_decoder_prompt_ids(language="zh", task="transcribe")

predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

print(text)