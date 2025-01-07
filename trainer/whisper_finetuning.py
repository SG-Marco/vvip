from datasets import load_dataset, DatasetDict

DATASET_ID = "Jzuluaga/atcosim_corpus" 
# or for UWB-ATCC corpus
# DATASET_ID = "Jzuluaga/uwb_atcc"

common_voice = DatasetDict()
common_voice["train"] = load_dataset(DATASET_ID, 'default', split="train")
common_voice["test"] = load_dataset(DATASET_ID, 'default', split="test")

# 2) 데이터셋 일부 샘플만 선택 
common_voice["train"] = common_voice["train"].select(range(10))
common_voice["test"] = common_voice["test"].select(range(10))

# audio와 lable text를 제외하고 모두 삭제
common_voice = common_voice.remove_columns(["id", "segment_start_time", "segment_end_time", "duration"])

# 'text' 컬럼을 'sentence'로 변경
common_voice = common_voice.rename_column("text", "sentence")


from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3", language="English", task="transcribe")
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3", language="English", task="transcribe")

def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=2)


from audio_prompting_whisper import AudioPromptingWhisper

# AudioPromptingWhisper 내부에서 WhisperForConditionalGeneration + AudioPrompter 생성
model = AudioPromptingWhisper(
    base_model_name="openai/whisper-large-v3",
    max_frames=3000,  # 필요에 맞춰 조정
    freeze_whisper=True   # Whisper 파라미터 동결, delta만 학습
)

model.whisper.generation_config.language = "english"
model.whisper.generation_config.task = "transcribe"
model.whisper.generation_config.forced_decoder_ids = None

import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
    

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.whisper.config.decoder_start_token_id,
)
    
import evaluate

metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-large-v3-en",  # change to a repo name of your choice
    use_safetensors=False,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=1,
    max_steps=5,
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1,
    eval_steps=1,
    logging_steps=1,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)

processor.save_pretrained(training_args.output_dir)

trainer.train()

# 학습 끝난 후 delta 저장
model.save_delta(path="audio_delta_final.pth")

print("Done!")